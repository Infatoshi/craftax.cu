"""Generate Craftax-Classic reference data for CUDA deterministic validation.

Runs the symbolic env for 100 steps with a fixed seed and fixed action sequence,
dumping map, observations, rewards, dones at each step.
"""
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams

def main():
    env = CraftaxClassicSymbolicEnv()
    params = env.default_params

    # Fixed seed
    rng = jax.random.PRNGKey(42)

    # Reset
    rng, reset_key = jax.random.split(rng)
    obs, state = env.reset(reset_key, params)

    # Extract initial map (64x64)
    initial_map = np.array(state.map, dtype=np.int8)

    # Action pattern: mix of movement and DO, repeating
    action_pattern = np.array([0,0,0,4,4,4,5,5,1,1,3,3,5,5,2,2,4,4], dtype=np.int32)

    N_STEPS = 100

    observations = []
    rewards = []
    dones = []
    maps = []
    actions_taken = []

    # Record initial obs
    observations.append(np.array(obs))
    maps.append(initial_map.copy())

    for step_i in range(N_STEPS):
        action = int(action_pattern[step_i % len(action_pattern)])
        actions_taken.append(action)

        rng, step_key = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_key, state, action, params)

        observations.append(np.array(obs))
        rewards.append(float(reward))
        dones.append(bool(done))
        maps.append(np.array(state.map, dtype=np.int8))

    # Save everything
    np.savez(
        "reference_data.npz",
        initial_map=initial_map,
        observations=np.stack(observations),        # (101, obs_dim) -- step 0 is reset obs
        rewards=np.array(rewards, dtype=np.float32), # (100,)
        dones=np.array(dones, dtype=np.bool_),       # (100,)
        maps=np.stack(maps),                          # (101, 64, 64)
        actions=np.array(actions_taken, dtype=np.int32), # (100,)
        # Also dump state_rng at step 0 for debugging
        state_rng_after_reset=np.array(state.state_rng if hasattr(state, 'state_rng') else [0,0], dtype=np.uint32),
        player_position_after_reset=np.array([32, 32], dtype=np.int32),
    )

    print(f"Saved reference_data.npz")
    print(f"  initial_map shape: {initial_map.shape}, dtype: {initial_map.dtype}")
    print(f"  observations shape: {np.stack(observations).shape}")
    print(f"  rewards: {np.array(rewards)[:10]}...")
    print(f"  dones: {np.array(dones)[:10]}...")
    print(f"  unique blocks in initial map: {np.unique(initial_map)}")
    print(f"  map[32,32] (player spawn): {initial_map[32,32]}")

if __name__ == "__main__":
    main()
