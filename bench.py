#!/usr/bin/env python3 -u
"""Benchmark CUDA Craftax-Classic: env-only SPS and full PPO training SPS."""
import torch
import torch.nn as nn
import time
import argparse
import craftax_cuda

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def make_policy(obs_dim, n_actions, hidden=64):
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions))
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))

        @torch.no_grad()
        def infer(self, obs):
            logits = self.actor(obs)
            values = self.critic(obs).squeeze(-1)
            u = torch.rand_like(logits).clamp_(1e-8, 1.0)
            actions = (logits - torch.log(-torch.log(u))).argmax(-1)
            lp = logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            return actions, lp, values

    return Policy().cuda()


def bench_env_only(num_envs_list, n_actions, warmup=10, iters=1000):
    """Benchmark env stepping without any policy inference."""
    print("\n--- ENV-ONLY SPS ---")
    for ne in num_envs_list:
        try:
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
            print(f"  NE={ne:>6}: {sps:>12,.0f} SPS")
            del env
        except Exception as e:
            print(f"  NE={ne:>6}: FAILED - {e}")


def bench_ppo(num_envs, num_steps, total_timesteps, hidden=64,
              lr=3e-4, gamma=0.99, gae_lambda=0.8, clip_eps=0.2,
              ent_coef=0.01, vf_coef=0.5, num_minibatches=4,
              update_epochs=1, max_grad_norm=0.5):
    """Benchmark full PPO training loop."""
    env = craftax_cuda.CraftaxEnv(num_envs, seed=42)
    obs_dim = env.get_obs_dim()
    n_actions = env.get_num_actions()
    num_updates = max(total_timesteps // (num_steps * num_envs), 1)
    mb_size = num_envs * num_steps // num_minibatches

    print(f"\n--- PPO TRAINING (NE={num_envs}, NS={num_steps}, updates={num_updates}) ---")

    policy = make_policy(obs_dim, n_actions, hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    obs_buf = torch.zeros(num_steps, num_envs, obs_dim, device='cuda')
    act_buf = torch.zeros(num_steps, num_envs, dtype=torch.int64, device='cuda')
    rew_buf = torch.zeros(num_steps, num_envs, device='cuda')
    done_buf = torch.zeros(num_steps, num_envs, device='cuda')
    val_buf = torch.zeros(num_steps, num_envs, device='cuda')
    logp_buf = torch.zeros(num_steps, num_envs, device='cuda')

    # Warmup
    obs = env.reset()
    for _ in range(3):
        a, lp, v = policy.infer(obs)
        obs, _, _ = env.step(a.int())
    obs = env.reset()
    torch.cuda.synchronize()

    t0 = time.time()
    total_steps = 0

    for update in range(num_updates):
        # Rollout
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

        # GAE
        with torch.no_grad():
            last_val = policy.critic(obs).squeeze(-1)
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

        # PPO update
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
                logits = policy.actor(b_obs[idx])
                val = policy.critic(b_obs[idx]).squeeze(-1)
                lp_all = logits.log_softmax(-1)
                nlp = lp_all.gather(-1, b_act[idx].unsqueeze(-1)).squeeze(-1)
                ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                ratio = (nlp - b_logp[idx]).exp()
                adv = b_adv[idx]
                pg = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()
                vc = b_val[idx] + (val - b_val[idx]).clamp(-clip_eps, clip_eps)
                vl = 0.5 * torch.max((val - b_ret[idx])**2, (vc - b_ret[idx])**2).mean()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--num-steps', type=int, default=64)
    parser.add_argument('--total-timesteps', type=int, default=10_000_000)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--num-minibatches', type=int, default=8)
    parser.add_argument('--env-only', action='store_true')
    args = parser.parse_args()

    env = craftax_cuda.CraftaxEnv(1, seed=0)
    n_actions = env.get_num_actions()
    del env

    bench_env_only([1024, 4096, 8192, 32768], n_actions)

    if not args.env_only:
        bench_ppo(args.num_envs, args.num_steps, args.total_timesteps,
                  hidden=args.hidden, update_epochs=args.update_epochs,
                  num_minibatches=args.num_minibatches)
