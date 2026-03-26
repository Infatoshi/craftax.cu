#!/usr/bin/env python3 -u
"""Render Craftax-Classic gameplay as a GIF from a trained PPO agent."""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import craftax_cuda
import os

torch.set_float32_matmul_precision('high')

# Block type -> RGB color (matches Craftax-Classic palette)
BLOCK_COLORS = {
    0:  (0, 0, 0),        # INVALID (black)
    1:  (40, 40, 40),     # OUT_OF_BOUNDS (dark gray)
    2:  (34, 139, 34),    # GRASS (forest green)
    3:  (30, 100, 200),   # WATER (blue)
    4:  (128, 128, 128),  # STONE (gray)
    5:  (0, 100, 0),      # TREE (dark green)
    6:  (139, 90, 43),    # WOOD (brown)
    7:  (194, 178, 128),  # PATH (tan)
    8:  (50, 50, 50),     # COAL (dark)
    9:  (180, 140, 100),  # IRON (rusty)
    10: (0, 255, 255),    # DIAMOND (cyan)
    11: (160, 82, 45),    # TABLE (sienna)
    12: (255, 69, 0),     # FURNACE (red-orange)
    13: (238, 214, 175),  # SAND (beige)
    14: (255, 50, 0),     # LAVA (bright red)
    15: (50, 205, 50),    # PLANT (lime green)
    16: (255, 215, 0),    # RIPE_PLANT (gold)
}

MOB_COLORS = {
    'zombie':   (100, 0, 150),   # purple
    'cow':      (200, 200, 200), # white
    'skeleton': (255, 255, 200), # pale yellow
    'arrow':    (255, 0, 0),     # red
}

PLAYER_COLOR = (255, 165, 0)  # orange

OBS_MAP_ROWS = 7
OBS_MAP_COLS = 9
OBS_MAP_CHANNELS = 21  # 17 block types + 4 mob types
N_MAP = OBS_MAP_ROWS * OBS_MAP_COLS * OBS_MAP_CHANNELS
CELL_SIZE = 12  # pixels per cell

# Inventory/stat names for the HUD
INV_NAMES = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling',
             'wpick', 'spick', 'ipick', 'wsword', 'ssword', 'isword']


def obs_to_frame(obs_np, step, ep_return):
    """Convert a single observation vector to an RGB image."""
    map_flat = obs_np[:N_MAP].reshape(OBS_MAP_ROWS, OBS_MAP_COLS, OBS_MAP_CHANNELS)
    block_map = map_flat[:, :, :17]  # one-hot block types
    mob_map = map_flat[:, :, 17:]    # zombie, cow, skeleton, arrow

    inv_start = N_MAP
    inv = obs_np[inv_start:inv_start + 12]
    intrinsics = obs_np[inv_start + 12:inv_start + 16]  # health, food, drink, energy
    direction = obs_np[inv_start + 16:inv_start + 20]
    light = obs_np[inv_start + 20]
    sleeping = obs_np[inv_start + 21]

    # Map image
    map_w = OBS_MAP_COLS * CELL_SIZE
    map_h = OBS_MAP_ROWS * CELL_SIZE
    hud_h = 60
    img = Image.new('RGB', (map_w, map_h + hud_h), (20, 20, 20))
    pixels = img.load()

    for r in range(OBS_MAP_ROWS):
        for c in range(OBS_MAP_COLS):
            block_type = int(block_map[r, c].argmax())
            color = BLOCK_COLORS.get(block_type, (0, 0, 0))

            # Apply light level
            lf = max(0.3, light)
            color = tuple(int(ch * lf) for ch in color)

            # Check for mobs
            has_zombie = mob_map[r, c, 0] > 0.5
            has_cow = mob_map[r, c, 1] > 0.5
            has_skel = mob_map[r, c, 2] > 0.5
            has_arrow = mob_map[r, c, 3] > 0.5

            for py in range(CELL_SIZE):
                for px in range(CELL_SIZE):
                    x = c * CELL_SIZE + px
                    y = r * CELL_SIZE + py
                    pixels[x, y] = color

            # Draw mob markers (small centered square)
            if has_zombie or has_cow or has_skel or has_arrow:
                mc = MOB_COLORS['zombie'] if has_zombie else \
                     MOB_COLORS['cow'] if has_cow else \
                     MOB_COLORS['skeleton'] if has_skel else \
                     MOB_COLORS['arrow']
                for py in range(3, CELL_SIZE - 3):
                    for px in range(3, CELL_SIZE - 3):
                        pixels[c * CELL_SIZE + px, r * CELL_SIZE + py] = mc

            # Player is always at center (3, 4)
            if r == 3 and c == 4:
                for py in range(2, CELL_SIZE - 2):
                    for px in range(2, CELL_SIZE - 2):
                        pixels[c * CELL_SIZE + px, r * CELL_SIZE + py] = PLAYER_COLOR

    # HUD
    # Draw stat bars at bottom
    bar_y = map_h + 5
    bar_h = 8
    bar_w = map_w - 10
    stat_names = ['HP', 'Food', 'Water', 'Energy']
    stat_colors = [(255, 50, 50), (200, 150, 50), (50, 100, 255), (50, 200, 50)]
    for i, (name, sc) in enumerate(zip(stat_names, stat_colors)):
        y = bar_y + i * (bar_h + 3)
        fill = int(intrinsics[i] * 10)  # 0-9 -> percentage
        fill_w = max(1, int(bar_w * fill / 9))
        for py in range(bar_h):
            for px in range(bar_w):
                x = 5 + px
                if x < 5 + fill_w:
                    pixels[x, y + py] = sc
                else:
                    pixels[x, y + py] = (40, 40, 40)

    # Scale up for visibility
    scale = 4
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def train_and_render(num_envs=256, train_steps=5_000_000, render_steps=500,
                     hidden=64, seed=42):
    """Train a quick agent, then render it playing."""
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

        if (update + 1) % 10 == 0:
            print(f"  Update {update+1}/{num_updates}", flush=True)

    print("Training done. Rendering...", flush=True)

    # Render with a single env
    render_env = craftax_cuda.CraftaxEnv(1, seed=99)
    obs = render_env.reset()
    frames = []
    ep_return = 0.0

    for step in range(render_steps):
        obs_np = obs[0].cpu().numpy()
        frame = obs_to_frame(obs_np, step, ep_return)
        frames.append(frame)

        with torch.no_grad():
            logits = policy.actor(obs)
            actions = logits.argmax(-1)  # greedy for rendering
        obs, rewards, dones = render_env.step(actions.int())
        ep_return += rewards[0].item()
        if dones[0]:
            ep_return = 0.0

    # Save GIF
    out_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(out_dir, 'gameplay.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=100, loop=0)
    print(f"Saved {gif_path} ({len(frames)} frames)")
    return gif_path


if __name__ == '__main__':
    train_and_render()
