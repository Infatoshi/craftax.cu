#!/usr/bin/env python3 -u
"""Minimal driver for nsys to measure host launch overhead."""
import argparse, torch
import craftax_cuda as oracle_mod
from experimental.build_opt import load_opt, load_opt5

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--which', choices=['oracle','opt','opt5'], default='opt5')
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--iters', type=int, default=500)
    a = p.parse_args()

    if a.which == 'oracle':
        env = oracle_mod.CraftaxEnv(a.num_envs, 42)
    elif a.which == 'opt':
        env = load_opt().CraftaxEnvOpt(a.num_envs, 42)
    else:
        env = load_opt5().CraftaxEnvOpt5(a.num_envs, 42)
    env.reset()
    torch.cuda.synchronize()
    act = torch.randint(0, env.get_num_actions(), (a.num_envs,), dtype=torch.int32, device='cuda')
    for _ in range(30): env.step(act)
    torch.cuda.synchronize()
    # The measurement region:
    torch.cuda.nvtx.range_push(f"{a.which}_hot_loop")
    for _ in range(a.iters):
        env.step(act)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
