#!/usr/bin/env python3 -u
"""Minimal driver to let NCU profile a small number of step kernel invocations."""
import argparse
import torch
import craftax_cuda as oracle_mod
from experimental.build_opt import load_opt


def run(which, num_envs, n_iters, n_warmup):
    if which == 'oracle':
        mod = oracle_mod
        cls = mod.CraftaxEnv
    else:
        mod = load_opt()
        cls = mod.CraftaxEnvOpt
    env = cls(num_envs, 42)
    env.reset()
    torch.cuda.synchronize()
    a = torch.randint(0, env.get_num_actions(), (num_envs,), dtype=torch.int32, device='cuda')
    for _ in range(n_warmup):
        env.step(a)
    torch.cuda.synchronize()
    # Named region so ncu -c <N> can target it precisely via --launch-skip
    for _ in range(n_iters):
        env.step(a)
    torch.cuda.synchronize()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--which', choices=['oracle', 'opt'], default='opt')
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--warmup', type=int, default=10)
    args = p.parse_args()
    run(args.which, args.num_envs, args.iters, args.warmup)
