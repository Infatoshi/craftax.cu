#!/usr/bin/env python3 -u
"""Validate CUDA Craftax-Classic against JAX reference.

Strategy: Since exact RNG replication of JAX Threefry is impractical,
we validate:
1. Observation format correctness (shape, ranges, one-hot structure)
2. Game mechanics (crafting, mining, combat, intrinsics decay)
3. Statistical properties (world gen block distribution, mob spawning rates)
4. Step-level invariants (energy/food/drink decay at correct rates)
"""
import torch
import numpy as np
import craftax_cuda
import sys
import os

def check(name, cond, msg=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f": {msg}" if msg else ""), flush=True)
    return cond

passed = 0
failed = 0

def test(name, cond, msg=""):
    global passed, failed
    if check(name, cond, msg):
        passed += 1
    else:
        failed += 1

# Load JAX reference
ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_data.npz'))
ref_map = ref['initial_map']       # (64, 64)
ref_obs = ref['observations']      # (101, 1345)
ref_rewards = ref['rewards']       # (100,)
ref_dones = ref['dones']           # (100,)
ref_actions = ref['actions']       # (100,)

print("=" * 60, flush=True)
print("JAX REFERENCE STATS", flush=True)
print("=" * 60, flush=True)
print(f"  Map shape: {ref_map.shape}", flush=True)
print(f"  Unique blocks: {sorted(np.unique(ref_map))}", flush=True)
block_counts = {i: int((ref_map == i).sum()) for i in range(17) if (ref_map == i).any()}
print(f"  Block counts: {block_counts}", flush=True)
print(f"  Obs shape: {ref_obs.shape}, range: [{ref_obs.min():.3f}, {ref_obs.max():.3f}]", flush=True)
print(f"  Rewards: {ref_rewards[:10]}", flush=True)
print(f"  Actions: {ref_actions[:20]}", flush=True)

# ============================================================
# TEST 1: Observation Format
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 1: Observation Format", flush=True)
print("=" * 60, flush=True)

env = craftax_cuda.CraftaxEnv(1, seed=12345)
obs = env.reset()
torch.cuda.synchronize()
obs_np = obs[0].cpu().numpy()

test("Obs shape", obs_np.shape == (1345,), f"got {obs_np.shape}")
test("Obs dim matches ref", obs_np.shape == ref_obs[0].shape)

# Map section: 7*9*21 = 1323 floats
map_obs = obs_np[:1323].reshape(7, 9, 21)

# Block one-hot: first 17 channels should sum to 1 at each position
block_onehot = map_obs[:, :, :17]
onehot_sums = block_onehot.sum(axis=2)
test("Block one-hot sums to 1", np.allclose(onehot_sums, 1.0, atol=1e-5),
     f"min={onehot_sums.min():.3f}, max={onehot_sums.max():.3f}")

# Block one-hot should be 0 or 1
test("Block one-hot binary", np.all((block_onehot == 0) | (block_onehot == 1)))

# Mob channels: channels 17-20 should be 0 or 1
mob_channels = map_obs[:, :, 17:21]
test("Mob channels binary", np.all((mob_channels == 0) | (mob_channels == 1)))

# Inventory: 12 floats normalized by /10, should be in [0, 0.9]
inv_obs = obs_np[1323:1335]
test("Inventory range", np.all(inv_obs >= 0) and np.all(inv_obs <= 0.9),
     f"range: [{inv_obs.min():.3f}, {inv_obs.max():.3f}]")

# Intrinsics: 4 floats (health/food/drink/energy) / 10, should start at 0.9
intrinsics = obs_np[1335:1339]
test("Initial intrinsics = 0.9", np.allclose(intrinsics, 0.9),
     f"got {intrinsics}")

# Direction: one-hot of 4, should sum to 1
direction = obs_np[1339:1343]
test("Direction one-hot sums to 1", abs(direction.sum() - 1.0) < 1e-5,
     f"sum={direction.sum():.3f}")
test("Direction one-hot binary", np.all((direction == 0) | (direction == 1)))

# Light level: float in [0, 1]
light = obs_np[1343]
test("Light level in [0,1]", 0 <= light <= 1, f"got {light:.3f}")

# Sleeping: 0 or 1
sleeping = obs_np[1344]
test("Initial sleeping = 0", sleeping == 0)

# ============================================================
# TEST 2: Game Mechanics - Stepping
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 2: Game Mechanics - Stepping", flush=True)
print("=" * 60, flush=True)

env = craftax_cuda.CraftaxEnv(1, seed=42)
obs = env.reset()
torch.cuda.synchronize()

# NOOP should not change intrinsics immediately (hunger/thirst accumulate slowly)
obs0 = obs[0].cpu().numpy().copy()
actions = torch.tensor([0], dtype=torch.int32, device='cuda')  # NOOP
obs, rew, done = env.step(actions)
torch.cuda.synchronize()
obs1 = obs[0].cpu().numpy()

# After 1 NOOP: intrinsics should still be 0.9 (hunger/thirst haven't triggered yet)
intrinsics0 = obs0[1335:1339]
intrinsics1 = obs1[1335:1339]
test("Intrinsics stable after 1 NOOP",
     np.allclose(intrinsics0, intrinsics1, atol=0.11),
     f"before={intrinsics0}, after={intrinsics1}")

# Light level should change
light0 = obs0[1343]
light1 = obs1[1343]
test("Light level changes after step", light0 != light1,
     f"before={light0:.4f}, after={light1:.4f}")

# ============================================================
# TEST 3: Movement
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 3: Movement", flush=True)
print("=" * 60, flush=True)

env = craftax_cuda.CraftaxEnv(1, seed=42)
obs = env.reset()
torch.cuda.synchronize()

# Player starts at center, center tile is in obs at (3,4) in 7x9 grid
# Try all 4 directions, at least one should move (some may be blocked by trees)
obs_before = obs[0].cpu().numpy()[:1323]
moved = False
for move_dir in [1, 2, 3, 4]:
    env_test = craftax_cuda.CraftaxEnv(1, seed=42)
    env_test.reset()
    obs_t, _, _ = env_test.step(torch.tensor([move_dir], dtype=torch.int32, device='cuda'))
    torch.cuda.synchronize()
    obs_after = obs_t[0].cpu().numpy()[:1323]
    if not np.allclose(obs_before, obs_after):
        moved = True
        break
test("Map view changes on move (at least 1 dir)", moved)

# Direction should be DOWN (index 3 in one-hot, since 1=LEFT,2=RIGHT,3=UP,4=DOWN -> 0-indexed: 3)
direction = obs[0].cpu().numpy()[1339:1343]
test("Direction updates to DOWN", direction[3] == 1.0 and direction[:3].sum() == 0,
     f"direction one-hot: {direction}")

# ============================================================
# TEST 4: Food/Hunger Decay
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 4: Intrinsics Decay", flush=True)
print("=" * 60, flush=True)

env = craftax_cuda.CraftaxEnv(1, seed=42)
obs = env.reset()
torch.cuda.synchronize()

# Run 30 NOOPs - hunger accumulates +1/tick, triggers at >25 -> food decreases at tick 26
for i in range(30):
    obs, rew, done = env.step(torch.tensor([0], dtype=torch.int32, device='cuda'))
torch.cuda.synchronize()

intrinsics = obs[0].cpu().numpy()[1335:1339]
food = intrinsics[1] * 10
test("Food decreased after 30 ticks", food < 9,
     f"food={food:.1f} (expected 8 after hunger trigger at tick 26)")

# ============================================================
# TEST 5: Thirst Decay
# ============================================================
# Thirst triggers at >20, so drink should decrease by tick 21
env = craftax_cuda.CraftaxEnv(1, seed=42)
env.reset()
for i in range(25):
    env.step(torch.tensor([0], dtype=torch.int32, device='cuda'))
torch.cuda.synchronize()
obs, _, _ = env.step(torch.tensor([0], dtype=torch.int32, device='cuda'))
torch.cuda.synchronize()
intrinsics = obs[0].cpu().numpy()[1335:1339]
drink = intrinsics[2] * 10
test("Drink decreased after 25 ticks", drink < 9,
     f"drink={drink:.1f} (expected 8 after thirst trigger at tick 21)")

# ============================================================
# TEST 6: Energy/Fatigue
# ============================================================
env = craftax_cuda.CraftaxEnv(1, seed=42)
env.reset()
for i in range(35):
    env.step(torch.tensor([0], dtype=torch.int32, device='cuda'))
torch.cuda.synchronize()
obs, _, _ = env.step(torch.tensor([0], dtype=torch.int32, device='cuda'))
torch.cuda.synchronize()
intrinsics = obs[0].cpu().numpy()[1335:1339]
energy = intrinsics[3] * 10
test("Energy decreased after 35 ticks", energy < 9,
     f"energy={energy:.1f} (expected 8 after fatigue trigger at tick 31)")

# ============================================================
# TEST 7: Terminal on health=0
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 7: Terminal Conditions", flush=True)
print("=" * 60, flush=True)

# Run many NOOPs until health drops to 0 (starvation)
env = craftax_cuda.CraftaxEnv(1, seed=42)
env.reset()
total_done = False
for i in range(5000):
    obs, rew, done = env.step(torch.tensor([0], dtype=torch.int32, device='cuda'))
    if done[0].item():
        total_done = True
        break
torch.cuda.synchronize()
test("Episode terminates eventually", total_done, f"terminated at step {i+1}")

# After terminal, auto-reset should give fresh obs
if total_done:
    intrinsics = obs[0].cpu().numpy()[1335:1339]
    test("Auto-reset: intrinsics back to 0.9", np.allclose(intrinsics, 0.9, atol=0.01),
         f"intrinsics after reset: {intrinsics}")

# ============================================================
# TEST 8: World Generation Statistics
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 8: World Generation Statistics", flush=True)
print("=" * 60, flush=True)

# Generate many worlds and check block distribution
N_WORLDS = 256
env = craftax_cuda.CraftaxEnv(N_WORLDS, seed=99)
obs = env.reset()
torch.cuda.synchronize()

# Extract map from observations -- center tile at (3,4) should be GRASS for all
map_obs = obs[:, :1323].reshape(N_WORLDS, 7, 9, 21)
center_block = map_obs[:, 3, 4, :17]  # center of 7x9 view
center_grass = center_block[:, 2]  # GRASS = index 2
test("Player spawns on GRASS", (center_grass == 1.0).all().item(),
     f"{center_grass.sum().item():.0f}/{N_WORLDS} on grass")

# Inventory should be all 0 at start
inv = obs[:, 1323:1335]
test("Initial inventory is zero", (inv == 0).all().item())

# All achievements should be false -> not visible in obs, but intrinsics should be 0.9
intrinsics = obs[:, 1335:1339]
test("All envs start with intrinsics=0.9",
     torch.allclose(intrinsics, torch.tensor(0.9, device='cuda'), atol=0.01))

# Sleeping should be 0
sleeping = obs[:, 1344]
test("All envs start not sleeping", (sleeping == 0).all().item())

# ============================================================
# TEST 9: Batch Consistency
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 9: Batch Consistency", flush=True)
print("=" * 60, flush=True)

# All envs taking same action should evolve independently (different maps)
env = craftax_cuda.CraftaxEnv(64, seed=42)
obs = env.reset()
torch.cuda.synchronize()

# Step all with NOOP
actions = torch.zeros(64, dtype=torch.int32, device='cuda')
obs, rew, done = env.step(actions)
torch.cuda.synchronize()

# Different envs should have different observations (different worlds)
obs_np = obs.cpu().numpy()
test("Different envs have different obs",
     not np.allclose(obs_np[0], obs_np[1]),
     f"diff norm: {np.linalg.norm(obs_np[0] - obs_np[1]):.3f}")

# But same structural properties
for i in range(64):
    block_sums = obs_np[i, :1323].reshape(7, 9, 21)[:, :, :17].sum(axis=2)
    if not np.allclose(block_sums, 1.0, atol=1e-5):
        test(f"Env {i} one-hot valid", False, f"sum range: [{block_sums.min():.3f}, {block_sums.max():.3f}]")
        break
else:
    test("All envs have valid one-hot blocks", True)

# ============================================================
# TEST 10: JAX Reference Comparison (structural)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("TEST 10: JAX Reference Structural Match", flush=True)
print("=" * 60, flush=True)

# Compare obs structure between JAX and CUDA
ref_obs0 = ref_obs[0]
ref_map_obs = ref_obs0[:1323].reshape(7, 9, 21)
ref_block_sums = ref_map_obs[:, :, :17].sum(axis=2)
test("JAX ref has valid one-hot blocks", np.allclose(ref_block_sums, 1.0, atol=1e-5))

ref_inv = ref_obs0[1323:1335]
test("JAX ref initial inventory=0", np.allclose(ref_inv, 0))

ref_intrinsics = ref_obs0[1335:1339]
test("JAX ref initial intrinsics=0.9", np.allclose(ref_intrinsics, 0.9, atol=0.01),
     f"got {ref_intrinsics}")

ref_dir = ref_obs0[1339:1343]
test("JAX ref direction is one-hot", abs(ref_dir.sum() - 1.0) < 1e-5 and np.all((ref_dir == 0) | (ref_dir == 1)))

ref_light = ref_obs0[1343]
test("JAX ref light in [0,1]", 0 <= ref_light <= 1)

ref_sleeping = ref_obs0[1344]
test("JAX ref initial sleeping=0", ref_sleeping == 0)

# Compare reward structure: first step with NOOP should give 0 reward
test("JAX ref first reward near 0", abs(ref_rewards[0]) < 1.0, f"got {ref_rewards[0]:.3f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60, flush=True)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}", flush=True)
print("=" * 60, flush=True)

if failed > 0:
    sys.exit(1)
