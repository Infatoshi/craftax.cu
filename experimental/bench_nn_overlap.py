#!/usr/bin/env python3 -u
"""Compare wall-clock SPS for env+NN loops:

  1. naive baseline   -- everything on default stream, strictly sequential
  2. two-stream pipe  -- env on env_stream, policy on nn_stream, CUDA events
                         synchronise exactly what's needed. Uses one-step
                         staleness between iterations to let env(k) run
                         concurrently with policy(k-1)'s forward/backward.
  3. multi-step chunks -- env.step_n(K) amortises launch; policy runs once
                          per K steps (stale actions within chunk).

Policy is a small MLP (nn.Linear -> cuBLAS). No cuDNN path because the
Craftax obs is already flat; a convolutional head would hit cuDNN.
"""
import argparse
import math
import time
import torch
import torch.nn as nn

import craftax_cuda as oracle_mod
from experimental.build_opt import load_opt5

torch.backends.cudnn.benchmark = True


CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS = 7, 9, 21
N_MAP = CRAFTAX_ROWS * CRAFTAX_COLS * CRAFTAX_CHANNELS
N_FLAT = 22


def mk_mlp(obs_dim, n_actions, hidden=128):
    # nn.Linear -> cuBLAS gemm. No cuDNN path since input is flat.
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_actions),
    ).cuda()


class PufferCNN(nn.Module):
    """Pufferlib Craftax arch: Conv2d -> Conv2d (cuDNN) + Linear branch (cuBLAS)."""
    def __init__(self, n_actions, cnn=32, hidden=128):
        super().__init__()
        self.me = nn.Sequential(
            nn.Conv2d(CRAFTAX_CHANNELS, cnn, 3, stride=2), nn.ReLU(),
            nn.Conv2d(cnn, cnn, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fe = nn.Sequential(nn.Linear(N_FLAT, hidden), nn.ReLU())
        self.pr = nn.Sequential(nn.Linear(2*cnn + hidden, hidden), nn.ReLU())
        self.a = nn.Linear(hidden, n_actions)

    def forward(self, obs):
        m = obs[:, :N_MAP].view(-1, CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS).permute(0, 3, 1, 2)
        m = self.me(m)
        f = self.fe(obs[:, N_MAP:])
        h = self.pr(torch.cat([m, f], dim=1))
        return self.a(h)


@torch.no_grad()
def sample_action(logits):
    # Gumbel-max sampler; deterministic & single cuBLAS + a few element-wise.
    u = torch.rand_like(logits).clamp_(1e-8, 1.0)
    return (logits - torch.log(-torch.log(u))).argmax(-1).to(torch.int32)


def bench_baseline(env, policy, n_steps):
    obs = env.reset()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        logits = policy(obs)
        actions = sample_action(logits)
        obs, _, _ = env.step(actions)
    torch.cuda.synchronize()
    return time.time() - t0


def bench_two_stream(env, policy, n_steps):
    """Overlap: policy.forward on stream B runs while env.step on stream A
    processes the PREVIOUS iteration's actions. Introduces a 1-step lag in
    the action/obs pairing, which is an acceptable latency for async RL."""
    env_stream = torch.cuda.Stream()
    nn_stream = torch.cuda.Stream()
    obs = env.reset()
    torch.cuda.synchronize()

    n_actions = env.get_num_actions()
    NE = obs.shape[0]

    # Prime: compute action_0 on nn_stream.
    with torch.cuda.stream(nn_stream):
        logits = policy(obs)
        actions = sample_action(logits)
    actions_ready = torch.cuda.Event()
    actions_ready.record(nn_stream)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(n_steps):
        # env_stream needs the actions computed by nn_stream
        env_stream.wait_event(actions_ready)
        with torch.cuda.stream(env_stream):
            obs, _, _ = env.step(actions)
        obs_ready = torch.cuda.Event()
        obs_ready.record(env_stream)

        # nn_stream starts forward on the newly-produced obs; this can overlap
        # with the NEXT env.step on env_stream if we kick it off here.
        nn_stream.wait_event(obs_ready)
        with torch.cuda.stream(nn_stream):
            logits = policy(obs)
            actions = sample_action(logits)
        actions_ready = torch.cuda.Event()
        actions_ready.record(nn_stream)

    torch.cuda.synchronize()
    return time.time() - t0


def bench_multistep_chunks(env, policy, n_steps, K):
    """env.step_n(K) runs K env-steps per launch (stale actions for last K-1
    of each chunk). Policy runs once per chunk. Demonstrates how amortising
    launches + reducing policy frequency trades exploration fidelity for SPS."""
    obs = env.reset()
    torch.cuda.synchronize()
    NE = obs.shape[0]
    n_chunks = n_steps // K
    t0 = time.time()
    for _ in range(n_chunks):
        # Single policy call produces K copies of the action.
        logits = policy(obs)
        a = sample_action(logits)                     # (NE,)
        actions = a.unsqueeze(0).expand(K, NE).contiguous()
        obs_seq, rew, done = env.step_n(actions)      # (K, NE, *)
        obs = obs_seq[-1]                             # next-iter obs = last one
    torch.cuda.synchronize()
    return time.time() - t0


def bench_multistep_pipelined(env, policy, n_steps, K):
    """Combines multi-step env kernel with stream overlap:
       env_stream.step_n(K) while nn_stream starts policy on the PREVIOUS
       chunk's final obs. Kernel time ~3ms dominates everything at K>=32 so
       the NN is fully hidden in the env kernel's shadow."""
    env_stream = torch.cuda.Stream()
    nn_stream = torch.cuda.Stream()
    obs = env.reset()
    torch.cuda.synchronize()
    NE = obs.shape[0]

    # Prime: one policy call on initial obs.
    with torch.cuda.stream(nn_stream):
        a = sample_action(policy(obs))
    a_ready = torch.cuda.Event(); a_ready.record(nn_stream)
    n_chunks = n_steps // K
    t0 = time.time()
    for _ in range(n_chunks):
        env_stream.wait_event(a_ready)
        with torch.cuda.stream(env_stream):
            actions = a.unsqueeze(0).expand(K, NE).contiguous()
            obs_seq, _, _ = env.step_n(actions)
            obs = obs_seq[-1]
        o_ready = torch.cuda.Event(); o_ready.record(env_stream)

        nn_stream.wait_event(o_ready)
        with torch.cuda.stream(nn_stream):
            a = sample_action(policy(obs))
        a_ready = torch.cuda.Event(); a_ready.record(nn_stream)
    torch.cuda.synchronize()
    return time.time() - t0


def bench_train_serial(env, policy, n_steps):
    """Minimal training loop: policy forward, fake loss (sum of logits^2 * reward
    as dummy signal), backward, optimizer. No streams. Measures full train SPS."""
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    obs = env.reset()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        logits = policy(obs)
        with torch.no_grad():
            a = sample_action(logits)
        obs_next, rew, _ = env.step(a)
        loss = (logits.pow(2).sum(-1) * rew).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        obs = obs_next
    torch.cuda.synchronize()
    return time.time() - t0


def bench_train_overlap(env, policy, n_steps):
    """Training with env on env_stream, NN forward/backward on nn_stream.
    Uses 1-step lag: env runs on previous actions while NN trains on previous obs."""
    env_stream = torch.cuda.Stream()
    nn_stream = torch.cuda.Stream()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    obs = env.reset()
    torch.cuda.synchronize()

    # Prime action
    with torch.cuda.stream(nn_stream):
        logits = policy(obs)
        a = sample_action(logits)
    a_ready = torch.cuda.Event(); a_ready.record(nn_stream)

    t0 = time.time()
    for _ in range(n_steps):
        env_stream.wait_event(a_ready)
        with torch.cuda.stream(env_stream):
            obs_next, rew, _ = env.step(a)
        env_ready = torch.cuda.Event(); env_ready.record(env_stream)

        nn_stream.wait_event(env_ready)
        with torch.cuda.stream(nn_stream):
            logits = policy(obs_next)  # fresh forward (with grad) for training
            loss = (logits.pow(2).sum(-1) * rew).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a = sample_action(logits)
        a_ready = torch.cuda.Event(); a_ready.record(nn_stream)
        obs = obs_next
    torch.cuda.synchronize()
    return time.time() - t0


def bench_cuda_graph(env, policy, n_steps, K):
    """CUDA-graph-captured multi-step chunk: one graph instance replays the
    entire (policy, sample, step_n) subgraph. Zero Python overhead per chunk."""
    obs = env.reset(); torch.cuda.synchronize()
    NE = obs.shape[0]
    n_actions = env.get_num_actions()

    # Pre-allocate static tensors the graph can reuse.
    static_obs_in = obs.clone()
    # Dummy warmup to materialise cuBLAS handles + any lazy allocation.
    g_stream = torch.cuda.Stream()
    s_out = torch.cuda.Stream()
    with torch.cuda.stream(g_stream):
        for _ in range(3):
            logits = policy(static_obs_in)
            a = sample_action(logits)
            actions = a.unsqueeze(0).expand(K, NE).contiguous()
            obs_seq, _, _ = env.step_n(actions)
            static_obs_in.copy_(obs_seq[-1])
    torch.cuda.current_stream().wait_stream(g_stream)
    torch.cuda.synchronize()

    # Capture the chunk.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(g_stream):
        with torch.cuda.graph(g):
            logits = policy(static_obs_in)
            a = sample_action(logits)
            actions = a.unsqueeze(0).expand(K, NE).contiguous()
            obs_seq, _, _ = env.step_n(actions)
            static_obs_in.copy_(obs_seq[-1])
    torch.cuda.synchronize()

    n_chunks = n_steps // K
    t0 = time.time()
    for _ in range(n_chunks):
        g.replay()
    torch.cuda.synchronize()
    return time.time() - t0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--n-steps', type=int, default=8192)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--which', choices=['oracle','opt5'], default='opt5')
    p.add_argument('--policy', choices=['mlp','cnn'], default='mlp')
    p.add_argument('--K', type=int, default=32)
    args = p.parse_args()

    if args.which == 'oracle':
        env = oracle_mod.CraftaxEnv(args.num_envs, 42)
        has_step_n = False
    else:
        m = load_opt5()
        env = m.CraftaxEnvOpt5(args.num_envs, 42)
        has_step_n = True

    obs_dim = env.get_obs_dim()
    n_actions = env.get_num_actions()
    if args.policy == 'mlp':
        policy = mk_mlp(obs_dim, n_actions, args.hidden)
    else:
        policy = PufferCNN(n_actions, hidden=args.hidden).cuda()
    n_params = sum(p.numel() for p in policy.parameters())
    kind = 'cuBLAS' if args.policy == 'mlp' else 'cuBLAS + cuDNN'
    print(f"Env: {args.which}, policy: {args.policy} ({kind}), NE={args.num_envs}, obs_dim={obs_dim}, hidden={args.hidden}, params={n_params:,}")

    # Warmup
    obs = env.reset()
    for _ in range(10):
        logits = policy(obs)
        a = sample_action(logits)
        obs, _, _ = env.step(a)
    torch.cuda.synchronize()

    def fmt(dt):
        sps = args.n_steps * args.num_envs / dt
        return f"{dt*1000:>7.1f} ms   {sps:>13,.0f} SPS"

    dt1 = bench_baseline(env, policy, args.n_steps)
    print(f"baseline (1 stream)     : {fmt(dt1)}")

    dt2 = bench_two_stream(env, policy, args.n_steps)
    print(f"2-stream pipelined      : {fmt(dt2)}   ({dt1/dt2:.2f}x vs baseline)")

    if has_step_n:
        dt3 = bench_multistep_chunks(env, policy, args.n_steps, args.K)
        print(f"multi-step K={args.K}, 1 stream: {fmt(dt3)}   ({dt1/dt3:.2f}x vs baseline)")

        dt4 = bench_multistep_pipelined(env, policy, args.n_steps, args.K)
        print(f"multi-step K={args.K}, piped  : {fmt(dt4)}   ({dt1/dt4:.2f}x vs baseline)")

        try:
            dt5 = bench_cuda_graph(env, policy, args.n_steps, args.K)
            print(f"CUDA graph  K={args.K}       : {fmt(dt5)}   ({dt1/dt5:.2f}x vs baseline)")
        except Exception as e:
            print(f"CUDA graph failed: {e}")

    # Training (forward + backward + optimizer)
    dt_t1 = bench_train_serial(env, policy, args.n_steps)
    print(f"train serial            : {fmt(dt_t1)}   (with backward + opt)")
    dt_t2 = bench_train_overlap(env, policy, args.n_steps)
    print(f"train 2-stream overlap  : {fmt(dt_t2)}   ({dt_t1/dt_t2:.2f}x vs train serial)")

    # Goal-oriented summary: time to reach 10M env-steps.
    target = 10_000_000
    print()
    print(f"Time to {target:,} env-steps:")
    print(f"  baseline (1 stream)     : {target/(args.n_steps*args.num_envs/dt1):>7.2f} s")
    print(f"  2-stream pipelined      : {target/(args.n_steps*args.num_envs/dt2):>7.2f} s")
    if has_step_n:
        print(f"  multi-step K={args.K}, 1 stream: {target/(args.n_steps*args.num_envs/dt3):>7.2f} s")
        print(f"  multi-step K={args.K}, piped  : {target/(args.n_steps*args.num_envs/dt4):>7.2f} s")
        try:
            print(f"  CUDA graph  K={args.K}       : {target/(args.n_steps*args.num_envs/dt5):>7.2f} s")
        except NameError:
            pass
