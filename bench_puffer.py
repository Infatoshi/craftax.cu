#!/usr/bin/env python3 -u
"""Benchmark craftax.cu with Pufferlib Craftax policy architecture.

Policy reference: pufferlib/environments/craftax/torch.py (PufferLib 3.0)
  map_encoder:  Conv2d(21, 32, 3, stride=2) -> ReLU -> Conv2d(32, 32, 3, stride=1) -> ReLU -> Flatten
                (input 7x9x21, output 64)
  flat_encoder: Linear(22, 128) -> ReLU
  proj:         Linear(64+128, 128) -> ReLU
  actor:        Linear(128, 17)
  value_fn:     Linear(128, 1)
"""
import torch
import torch.nn as nn
import time
import argparse
import math
import craftax_cuda

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CRAFTAX_ROWS = 7
CRAFTAX_COLS = 9
CRAFTAX_CHANNELS = 21
N_MAP = CRAFTAX_ROWS * CRAFTAX_COLS * CRAFTAX_CHANNELS  # 1323
N_FLAT = 22


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PufferPolicy(nn.Module):
    def __init__(self, n_actions, cnn_channels=32, hidden_size=128):
        super().__init__()
        self.map_encoder = nn.Sequential(
            layer_init(nn.Conv2d(CRAFTAX_CHANNELS, cnn_channels, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flat_encoder = nn.Sequential(
            layer_init(nn.Linear(N_FLAT, hidden_size)),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            layer_init(nn.Linear(2 * cnn_channels + hidden_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)
        self.value_fn = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def encode(self, obs):
        map_obs = obs[:, :N_MAP].view(-1, CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS).permute(0, 3, 1, 2)
        m = self.map_encoder(map_obs)
        f = self.flat_encoder(obs[:, N_MAP:])
        return self.proj(torch.cat([m, f], dim=1))

    def forward(self, obs):
        h = self.encode(obs)
        return self.actor(h), self.value_fn(h).squeeze(-1)

    @torch.no_grad()
    def infer(self, obs):
        logits, values = self.forward(obs)
        u = torch.rand_like(logits).clamp_(1e-8, 1.0)
        actions = (logits - torch.log(-torch.log(u))).argmax(-1)
        lp = logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, lp, values


def bench_env_only(num_envs_list, n_actions, warmup=10, iters=1000):
    print("\n--- ENV-ONLY SPS ---")
    results = {}
    for ne in num_envs_list:
        env = craftax_cuda.CraftaxEnv(ne, seed=42)
        env.reset()
        torch.cuda.synchronize()
        actions = torch.randint(0, n_actions, (ne,), dtype=torch.int32, device='cuda')
        for _ in range(warmup):
            env.step(actions)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            env.step(actions)
        torch.cuda.synchronize()
        sps = ne * iters / (time.time() - t0)
        results[ne] = sps
        print(f"  NE={ne:>6}: {sps:>12,.0f} SPS")
        del env
    return results


def bench_ppo(num_envs, num_steps, total_timesteps, hidden_size=128, cnn_channels=32,
              lr=3e-4, gamma=0.99, gae_lambda=0.8, clip_eps=0.2,
              ent_coef=0.01, vf_coef=0.5, num_minibatches=8,
              update_epochs=4, max_grad_norm=0.5):
    env = craftax_cuda.CraftaxEnv(num_envs, seed=42)
    obs_dim = env.get_obs_dim()
    n_actions = env.get_num_actions()
    assert obs_dim == N_MAP + N_FLAT, f"obs_dim {obs_dim} != {N_MAP + N_FLAT}"
    num_updates = max(total_timesteps // (num_steps * num_envs), 1)
    mb_size = num_envs * num_steps // num_minibatches

    print(f"\n--- PPO TRAINING (Pufferlib arch: cnn={cnn_channels}, hidden={hidden_size}) ---")
    print(f"  NE={num_envs}, NS={num_steps}, updates={num_updates}, epochs={update_epochs}, nmb={num_minibatches}")

    policy = PufferPolicy(n_actions, cnn_channels, hidden_size).cuda()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Params: {n_params:,}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    obs_buf = torch.zeros(num_steps, num_envs, obs_dim, device='cuda')
    act_buf = torch.zeros(num_steps, num_envs, dtype=torch.int64, device='cuda')
    rew_buf = torch.zeros(num_steps, num_envs, device='cuda')
    done_buf = torch.zeros(num_steps, num_envs, device='cuda')
    val_buf = torch.zeros(num_steps, num_envs, device='cuda')
    logp_buf = torch.zeros(num_steps, num_envs, device='cuda')

    obs = env.reset()
    for _ in range(3):
        a, lp, v = policy.infer(obs)
        obs, _, _ = env.step(a.int())
    obs = env.reset()
    torch.cuda.synchronize()

    t0 = time.time()
    total_steps = 0
    for update in range(num_updates):
        for step in range(num_steps):
            actions, logprobs, values = policy.infer(obs)
            obs_buf[step] = obs
            act_buf[step] = actions
            val_buf[step] = values
            logp_buf[step] = logprobs
            obs, rewards, dones = env.step(actions.int())
            rew_buf[step] = rewards
            done_buf[step] = dones.float()
        total_steps += num_steps * num_envs

        with torch.no_grad():
            _, last_val = policy.forward(obs)
            not_dones = 1.0 - done_buf
            nv = torch.empty_like(val_buf)
            nv[:-1] = val_buf[1:]
            nv[-1] = last_val
            deltas = rew_buf + gamma * nv * not_dones - val_buf
            adv_acc = torch.zeros(num_envs, device='cuda')
            returns = torch.zeros(num_steps, num_envs, device='cuda')
            for t in range(num_steps - 1, -1, -1):
                adv_acc = deltas[t] + gamma * gae_lambda * not_dones[t] * adv_acc
                returns[t] = adv_acc + val_buf[t]

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)
        b_adv = (returns - val_buf).reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        bs = num_envs * num_steps

        for epoch in range(update_epochs):
            perm = torch.randperm(bs, device='cuda')
            for start in range(0, bs, mb_size):
                idx = perm[start:start + mb_size]
                logits, val = policy.forward(b_obs[idx])
                lp_all = logits.log_softmax(-1)
                nlp = lp_all.gather(-1, b_act[idx].unsqueeze(-1)).squeeze(-1)
                ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                ratio = (nlp - b_logp[idx]).exp()
                adv = b_adv[idx]
                pg = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()
                vc = b_val[idx] + (val - b_val[idx]).clamp(-clip_eps, clip_eps)
                vl = 0.5 * torch.max((val - b_ret[idx]) ** 2, (vc - b_ret[idx]) ** 2).mean()
                loss = pg + vf_coef * vl - ent_coef * ent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    sps = total_steps / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Total: {total_steps:,} steps in {elapsed:.2f}s")
    print(f"  Training SPS: {sps:,.0f}")
    print(f"  Peak GPU memory: {peak_gb:.2f} GB")
    return sps


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--num-steps', type=int, default=64)
    p.add_argument('--total-timesteps', type=int, default=10_000_000)
    p.add_argument('--hidden-size', type=int, default=128)
    p.add_argument('--cnn-channels', type=int, default=32)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--num-minibatches', type=int, default=8)
    p.add_argument('--env-only', action='store_true')
    p.add_argument('--sweep', action='store_true')
    args = p.parse_args()

    env = craftax_cuda.CraftaxEnv(1, seed=0)
    n_actions = env.get_num_actions()
    del env

    bench_env_only([1024, 4096, 8192, 32768, 65536], n_actions)

    if args.env_only:
        raise SystemExit(0)

    if args.sweep:
        for ne, ns in [(4096, 64), (8192, 64), (16384, 32), (32768, 32)]:
            bench_ppo(ne, ns, args.total_timesteps,
                      hidden_size=args.hidden_size, cnn_channels=args.cnn_channels,
                      update_epochs=args.update_epochs,
                      num_minibatches=args.num_minibatches)
    else:
        bench_ppo(args.num_envs, args.num_steps, args.total_timesteps,
                  hidden_size=args.hidden_size, cnn_channels=args.cnn_channels,
                  update_epochs=args.update_epochs,
                  num_minibatches=args.num_minibatches)
