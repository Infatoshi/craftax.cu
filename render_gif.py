#!/usr/bin/env python3 -u
"""Render Craftax-Classic gameplay as a GIF using the official JAX pixel renderer.

Train a quick agent with the CUDA env, then replay with the JAX env to get
proper textured renders.
"""
import torch
import torch.nn as nn
import numpy as np
import os

# ---- Phase 1: Train a policy on CUDA env ----
def train_policy(num_envs=256, train_steps=5_000_000, hidden=64, seed=42):
    import craftax_cuda

    env = craftax_cuda.CraftaxEnv(num_envs, seed=seed)
    obs_dim = env.get_obs_dim()
    n_actions = env.get_num_actions()

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

    policy = Policy().cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    ns = 64
    gamma, gae_lam, clip_eps = 0.99, 0.8, 0.2
    num_updates = train_steps // (ns * num_envs)

    obs_buf = torch.zeros(ns, num_envs, obs_dim, device='cuda')
    act_buf = torch.zeros(ns, num_envs, dtype=torch.int64, device='cuda')
    rew_buf = torch.zeros(ns, num_envs, device='cuda')
    done_buf = torch.zeros(ns, num_envs, device='cuda')
    val_buf = torch.zeros(ns, num_envs, device='cuda')
    logp_buf = torch.zeros(ns, num_envs, device='cuda')

    obs = env.reset()
    print(f"Training for {num_updates} updates ({train_steps:,} steps)...", flush=True)

    for update in range(num_updates):
        for step in range(ns):
            actions, logprobs, values = policy.infer(obs)
            obs_buf[step] = obs; act_buf[step] = actions
            val_buf[step] = values; logp_buf[step] = logprobs
            obs, rewards, dones = env.step(actions.int())
            rew_buf[step] = rewards; done_buf[step] = dones.float()

        with torch.no_grad():
            last_val = policy.critic(obs).squeeze(-1)
            nd = 1.0 - done_buf
            nv = torch.empty_like(val_buf)
            nv[:-1] = val_buf[1:]; nv[-1] = last_val
            deltas = rew_buf + gamma * nv * nd - val_buf
            aa = torch.zeros(num_envs, device='cuda')
            returns = torch.zeros(ns, num_envs, device='cuda')
            for t in range(ns - 1, -1, -1):
                aa = deltas[t] + gamma * gae_lam * nd[t] * aa
                returns[t] = aa + val_buf[t]

        bo = obs_buf.reshape(-1, obs_dim); ba = act_buf.reshape(-1)
        bl = logp_buf.reshape(-1); br = returns.reshape(-1); bv = val_buf.reshape(-1)
        bad = (returns - val_buf).reshape(-1)
        bad = (bad - bad.mean()) / (bad.std() + 1e-8)
        bs = num_envs * ns; mb = bs // 4
        perm = torch.randperm(bs, device='cuda')
        for start in range(0, bs, mb):
            idx = perm[start:start + mb]
            lo = policy.actor(bo[idx]); va = policy.critic(bo[idx]).squeeze(-1)
            la = lo.log_softmax(-1)
            nlp = la.gather(-1, ba[idx].unsqueeze(-1)).squeeze(-1)
            ent = -(la.exp() * la).sum(-1).mean()
            r = (nlp - bl[idx]).exp(); ad = bad[idx]
            pg = -torch.min(r * ad, r.clamp(1 - clip_eps, 1 + clip_eps) * ad).mean()
            vc = bv[idx] + (va - bv[idx]).clamp(-clip_eps, clip_eps)
            vl = 0.5 * torch.max((va - br[idx])**2, (vc - br[idx])**2).mean()
            loss = pg + 0.5 * vl - 0.01 * ent
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (update + 1) % 50 == 0:
            print(f"  Update {update+1}/{num_updates}", flush=True)

    print("Training done.", flush=True)
    return policy


# ---- Phase 2: Render with JAX pixel renderer ----
def render_with_jax(policy, render_steps=150, block_pixel_size=16, seed=99):
    import jax
    import jax.numpy as jnp
    from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
    from craftax.craftax_classic.renderer import render_craftax_pixels
    from PIL import Image

    env = CraftaxClassicSymbolicEnv()
    params = env.default_params
    rng = jax.random.PRNGKey(seed)

    rng, reset_key = jax.random.split(rng)
    obs_jax, state = env.reset(reset_key, params)

    frames = []
    ep_return = 0.0

    print(f"Rendering {render_steps} frames...", flush=True)
    for step in range(render_steps):
        # Render the current state with the official pixel renderer
        pixels = render_craftax_pixels(state, block_pixel_size)
        pixels_np = np.array(pixels)
        pixels_uint8 = np.clip(pixels_np, 0, 255).astype(np.uint8)
        frame = Image.fromarray(pixels_uint8)
        frames.append(frame)

        # Get action from trained CUDA policy
        obs_np = np.array(obs_jax)
        obs_torch = torch.from_numpy(obs_np).float().unsqueeze(0).cuda()
        with torch.no_grad():
            logits = policy.actor(obs_torch)
            action = logits.argmax(-1).item()

        # Step JAX env
        rng, step_key = jax.random.split(rng)
        obs_jax, state, reward, done, info = env.step(step_key, state, action, params)
        ep_return += float(reward)

        if done:
            rng, reset_key = jax.random.split(rng)
            obs_jax, state = env.reset(reset_key, params)
            ep_return = 0.0

        if (step + 1) % 100 == 0:
            print(f"  Frame {step+1}/{render_steps}", flush=True)

    # Save GIF
    out_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(out_dir, 'gameplay.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=100, loop=0)
    print(f"Saved {gif_path} ({len(frames)} frames, {os.path.getsize(gif_path)/1024:.0f} KB)")
    return gif_path


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    policy = train_policy()
    render_with_jax(policy)
