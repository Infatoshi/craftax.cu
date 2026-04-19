#!/usr/bin/env python3 -u
import argparse, torch
import craftax_cuda as oracle_mod
from experimental.build_opt import load_opt2

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=32768)
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--block', type=int, default=256)
    a = p.parse_args()
    m = load_opt2()
    env = m.CraftaxEnvOpt2(a.num_envs, 42, a.block)
    env.reset(); torch.cuda.synchronize()
    act = torch.randint(0, env.get_num_actions(), (a.num_envs,), dtype=torch.int32, device='cuda')
    for _ in range(a.warmup): env.step(act)
    torch.cuda.synchronize()
    for _ in range(a.iters): env.step(act)
    torch.cuda.synchronize()
