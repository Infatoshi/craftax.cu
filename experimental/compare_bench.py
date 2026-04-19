#!/usr/bin/env python3 -u
"""Oracle vs Opt: step-by-step parity + env-only SPS + PPO training SPS (Pufferlib arch)."""
import argparse
import math
import time
import torch
import torch.nn as nn

import craftax_cuda as oracle_mod
from experimental.build_opt import load_opt, load_opt2

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS = 7, 9, 21
N_MAP = CRAFTAX_ROWS * CRAFTAX_COLS * CRAFTAX_CHANNELS
N_FLAT = 22


def layer_init(l, std=math.sqrt(2), b=0.0):
    nn.init.orthogonal_(l.weight, std)
    nn.init.constant_(l.bias, b)
    return l


class PufferPolicy(nn.Module):
    def __init__(self, n_actions, cnn_channels=32, hidden_size=128):
        super().__init__()
        self.map_encoder = nn.Sequential(
            layer_init(nn.Conv2d(CRAFTAX_CHANNELS, cnn_channels, 3, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)), nn.ReLU(),
            nn.Flatten())
        self.flat_encoder = nn.Sequential(layer_init(nn.Linear(N_FLAT, hidden_size)), nn.ReLU())
        self.proj = nn.Sequential(
            layer_init(nn.Linear(2 * cnn_channels + hidden_size, hidden_size)), nn.ReLU())
        self.actor = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)
        self.value_fn = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, obs):
        m = obs[:, :N_MAP].view(-1, CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS).permute(0, 3, 1, 2)
        m = self.map_encoder(m)
        f = self.flat_encoder(obs[:, N_MAP:])
        h = self.proj(torch.cat([m, f], dim=1))
        return self.actor(h), self.value_fn(h).squeeze(-1)

    @torch.no_grad()
    def infer(self, obs):
        logits, values = self.forward(obs)
        u = torch.rand_like(logits).clamp_(1e-8, 1.0)
        actions = (logits - torch.log(-torch.log(u))).argmax(-1)
        lp = logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, lp, values


def verify_parity(num_envs=128, num_steps=500, seed=42, variant='opt'):
    print(f"\n=== PARITY [{variant}]: NE={num_envs}, steps={num_steps}, seed={seed} ===")
    if variant == 'opt':
        opt_mod = load_opt()
        env_x = opt_mod.CraftaxEnvOpt(num_envs, seed)
    else:
        opt_mod = load_opt2()
        env_x = opt_mod.CraftaxEnvOpt2(num_envs, seed, 64)
    env_o = oracle_mod.CraftaxEnv(num_envs, seed)

    o_o = env_o.reset()
    o_x = env_x.reset()
    torch.cuda.synchronize()

    assert torch.equal(o_o, o_x), f"reset obs diverge: max|d|={((o_o - o_x).abs()).max().item()}"
    print(f"  reset obs: bitwise equal ({o_o.shape})")

    gen = torch.Generator(device='cuda').manual_seed(123)
    n_actions = env_o.get_num_actions()
    total_reward_diff = 0.0
    total_done_diff = 0
    max_obs_diff = 0.0
    for t in range(num_steps):
        a = torch.randint(0, n_actions, (num_envs,), dtype=torch.int32, device='cuda', generator=gen)
        obs_o, rew_o, don_o = env_o.step(a)
        obs_x, rew_x, don_x = env_x.step(a)
        torch.cuda.synchronize()
        if not torch.equal(obs_o, obs_x):
            d = (obs_o - obs_x).abs().max().item()
            max_obs_diff = max(max_obs_diff, d)
            print(f"  FAIL step {t}: obs max|d|={d}")
            # locate the env
            env_idx = (obs_o - obs_x).abs().sum(-1).argmax().item()
            print(f"    worst env: {env_idx}")
            return False
        if not torch.equal(rew_o, rew_x):
            d = (rew_o - rew_x).abs().max().item()
            total_reward_diff += d
            print(f"  FAIL step {t}: reward max|d|={d}")
            return False
        if not torch.equal(don_o, don_x):
            total_done_diff += int((don_o != don_x).sum().item())
            print(f"  FAIL step {t}: dones differ ({(don_o != don_x).sum().item()} envs)")
            return False
    print(f"  all {num_steps} steps bitwise equal (obs, rewards, dones)")
    del env_o, env_x
    return True


def bench_env_only(mod, name, num_envs_list, n_actions, warmup=10, iters=1000):
    print(f"\n--- ENV-ONLY SPS [{name}] ---")
    cls = getattr(mod, 'CraftaxEnv', None) or getattr(mod, 'CraftaxEnvOpt', None) or mod.CraftaxEnvOpt2
    results = {}
    for ne in num_envs_list:
        env = cls(ne, 42)
        env.reset()
        torch.cuda.synchronize()
        a = torch.randint(0, n_actions, (ne,), dtype=torch.int32, device='cuda')
        for _ in range(warmup):
            env.step(a)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            env.step(a)
        torch.cuda.synchronize()
        sps = ne * iters / (time.time() - t0)
        results[ne] = sps
        print(f"  NE={ne:>6}: {sps:>12,.0f} SPS")
        del env
    return results


def bench_ppo(mod, name, num_envs, num_steps, total_timesteps,
              hidden_size=128, cnn_channels=32, lr=3e-4, gamma=0.99, gae_lambda=0.8,
              clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, num_minibatches=8,
              update_epochs=4, max_grad_norm=0.5):
    cls = getattr(mod, 'CraftaxEnv', None) or getattr(mod, 'CraftaxEnvOpt', None) or mod.CraftaxEnvOpt2
    env = cls(num_envs, 42)
    obs_dim = env.get_obs_dim()
    n_actions = env.get_num_actions()
    num_updates = max(total_timesteps // (num_steps * num_envs), 1)
    mb_size = num_envs * num_steps // num_minibatches
    print(f"\n--- PPO [{name}]  NE={num_envs}, NS={num_steps}, upd={num_updates}, ep={update_epochs}, mb={num_minibatches} ---")

    policy = PufferPolicy(n_actions, cnn_channels, hidden_size).cuda()
    opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    obs_buf = torch.zeros(num_steps, num_envs, obs_dim, device='cuda')
    act_buf = torch.zeros(num_steps, num_envs, dtype=torch.int64, device='cuda')
    rew_buf = torch.zeros(num_steps, num_envs, device='cuda')
    don_buf = torch.zeros(num_steps, num_envs, device='cuda')
    val_buf = torch.zeros(num_steps, num_envs, device='cuda')
    lpf_buf = torch.zeros(num_steps, num_envs, device='cuda')

    obs = env.reset()
    for _ in range(3):
        a, lp, v = policy.infer(obs)
        obs, _, _ = env.step(a.int())
    obs = env.reset()
    torch.cuda.synchronize()

    t0 = time.time()
    total = 0
    for _ in range(num_updates):
        for s in range(num_steps):
            a, lp, v = policy.infer(obs)
            obs_buf[s] = obs; act_buf[s] = a; val_buf[s] = v; lpf_buf[s] = lp
            obs, r, d = env.step(a.int())
            rew_buf[s] = r; don_buf[s] = d.float()
        total += num_steps * num_envs

        with torch.no_grad():
            _, last_v = policy.forward(obs)
            nd = 1.0 - don_buf
            nv = torch.empty_like(val_buf)
            nv[:-1] = val_buf[1:]; nv[-1] = last_v
            deltas = rew_buf + gamma * nv * nd - val_buf
            acc = torch.zeros(num_envs, device='cuda')
            rets = torch.zeros_like(val_buf)
            for t in range(num_steps - 1, -1, -1):
                acc = deltas[t] + gamma * gae_lambda * nd[t] * acc
                rets[t] = acc + val_buf[t]

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_lp = lpf_buf.reshape(-1)
        b_ret = rets.reshape(-1)
        b_val = val_buf.reshape(-1)
        b_adv = (rets - val_buf).reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        bs = num_envs * num_steps

        for _ in range(update_epochs):
            perm = torch.randperm(bs, device='cuda')
            for start in range(0, bs, mb_size):
                idx = perm[start:start + mb_size]
                lg, val = policy.forward(b_obs[idx])
                lpa = lg.log_softmax(-1)
                nlp = lpa.gather(-1, b_act[idx].unsqueeze(-1)).squeeze(-1)
                ent = -(lpa.exp() * lpa).sum(-1).mean()
                ratio = (nlp - b_lp[idx]).exp()
                adv = b_adv[idx]
                pg = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()
                vc = b_val[idx] + (val - b_val[idx]).clamp(-clip_eps, clip_eps)
                vl = 0.5 * torch.max((val - b_ret[idx])**2, (vc - b_ret[idx])**2).mean()
                loss = pg + vf_coef * vl - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt.step()

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    sps = total / elapsed
    print(f"  {total:,} steps in {elapsed:.2f}s -> {sps:,.0f} SPS  (peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")
    del env, policy, opt
    torch.cuda.empty_cache()
    return sps


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--skip-parity', action='store_true')
    p.add_argument('--parity-envs', type=int, default=128)
    p.add_argument('--parity-steps', type=int, default=500)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--num-steps', type=int, default=64)
    p.add_argument('--total-timesteps', type=int, default=10_000_000)
    p.add_argument('--env-only', action='store_true')
    args = p.parse_args()

    opt_mod = load_opt()

    if not args.skip_parity:
        ok = verify_parity(args.parity_envs, args.parity_steps)
        if not ok:
            raise SystemExit(1)

    env = oracle_mod.CraftaxEnv(1, 0)
    n_actions = env.get_num_actions()
    del env

    ne_list = [1024, 4096, 8192, 32768, 65536]
    r_oracle = bench_env_only(oracle_mod, 'ORACLE', ne_list, n_actions)
    torch.cuda.empty_cache()
    r_opt = bench_env_only(opt_mod, 'OPT', ne_list, n_actions)

    print("\n--- env-only speedup (OPT / ORACLE) ---")
    for ne in ne_list:
        print(f"  NE={ne:>6}: {r_opt[ne]/r_oracle[ne]:.3f}x  "
              f"({r_oracle[ne]:>12,.0f} -> {r_opt[ne]:>12,.0f})")

    if args.env_only:
        raise SystemExit(0)

    t_oracle = bench_ppo(oracle_mod, 'ORACLE', args.num_envs, args.num_steps, args.total_timesteps)
    t_opt = bench_ppo(opt_mod, 'OPT', args.num_envs, args.num_steps, args.total_timesteps)
    print(f"\n--- training speedup (OPT / ORACLE): {t_opt/t_oracle:.3f}x  "
          f"({t_oracle:,.0f} -> {t_opt:,.0f}) ---")
