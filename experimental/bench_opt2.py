#!/usr/bin/env python3 -u
"""Bench oracle vs opt vs opt2, env-only SPS + PPO (Pufferlib arch)."""
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


def li(l, std=math.sqrt(2), b=0.0):
    nn.init.orthogonal_(l.weight, std); nn.init.constant_(l.bias, b); return l

class PufferPolicy(nn.Module):
    def __init__(self, n_actions, cnn=32, hid=128):
        super().__init__()
        self.me = nn.Sequential(li(nn.Conv2d(CRAFTAX_CHANNELS, cnn, 3, stride=2)), nn.ReLU(),
                                li(nn.Conv2d(cnn, cnn, 3, stride=1)), nn.ReLU(), nn.Flatten())
        self.fe = nn.Sequential(li(nn.Linear(N_FLAT, hid)), nn.ReLU())
        self.pr = nn.Sequential(li(nn.Linear(2*cnn + hid, hid)), nn.ReLU())
        self.a = li(nn.Linear(hid, n_actions), std=0.01)
        self.v = li(nn.Linear(hid, 1), std=1.0)
    def forward(self, o):
        m = o[:, :N_MAP].view(-1, CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS).permute(0,3,1,2)
        m = self.me(m); f = self.fe(o[:, N_MAP:])
        h = self.pr(torch.cat([m, f], 1))
        return self.a(h), self.v(h).squeeze(-1)
    @torch.no_grad()
    def infer(self, o):
        lg, v = self.forward(o)
        u = torch.rand_like(lg).clamp_(1e-8, 1.0)
        a = (lg - torch.log(-torch.log(u))).argmax(-1)
        lp = lg.log_softmax(-1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        return a, lp, v


def make_env(which, NE, seed=42, block=64):
    if which == 'oracle':
        return oracle_mod.CraftaxEnv(NE, seed)
    if which == 'opt':
        return load_opt().CraftaxEnvOpt(NE, seed)
    return load_opt2().CraftaxEnvOpt2(NE, seed, block)


def bench_env_only(which, ne_list, n_actions, warmup=10, iters=1000, block=64):
    print(f"\n--- ENV-ONLY [{which}] (block={block if which=='opt2' else 256}) ---")
    out = {}
    for ne in ne_list:
        env = make_env(which, ne, block=block)
        env.reset()
        torch.cuda.synchronize()
        a = torch.randint(0, n_actions, (ne,), dtype=torch.int32, device='cuda')
        for _ in range(warmup): env.step(a)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters): env.step(a)
        torch.cuda.synchronize()
        sps = ne * iters / (time.time() - t0)
        out[ne] = sps
        print(f"  NE={ne:>6}: {sps:>12,.0f} SPS")
        del env
    return out


def bench_ppo(which, NE, NS, total, block=64, hid=128, cnn=32, lr=3e-4,
              gamma=0.99, lam=0.8, clip=0.2, ec=0.01, vc=0.5, nmb=8, ep=4, mg=0.5):
    env = make_env(which, NE, block=block)
    obs_dim = env.get_obs_dim(); n_act = env.get_num_actions()
    nupd = max(total // (NS * NE), 1); mbs = NE * NS // nmb
    print(f"\n--- PPO [{which}] NE={NE} NS={NS} upd={nupd} ep={ep} mb={nmb} ---")
    pol = PufferPolicy(n_act, cnn, hid).cuda()
    opt = torch.optim.Adam(pol.parameters(), lr=lr, eps=1e-5)
    ob = torch.zeros(NS, NE, obs_dim, device='cuda')
    ac = torch.zeros(NS, NE, dtype=torch.int64, device='cuda')
    rw = torch.zeros(NS, NE, device='cuda'); dn = torch.zeros(NS, NE, device='cuda')
    vl = torch.zeros(NS, NE, device='cuda'); lp = torch.zeros(NS, NE, device='cuda')
    obs = env.reset()
    for _ in range(3):
        a, l, v = pol.infer(obs); obs, _, _ = env.step(a.int())
    obs = env.reset(); torch.cuda.synchronize()
    t0 = time.time(); tot = 0
    for _ in range(nupd):
        for s in range(NS):
            a, l, v = pol.infer(obs)
            ob[s]=obs; ac[s]=a; vl[s]=v; lp[s]=l
            obs, r, d = env.step(a.int()); rw[s]=r; dn[s]=d.float()
        tot += NS * NE
        with torch.no_grad():
            _, lv = pol.forward(obs)
            nd = 1.0 - dn; nv = torch.empty_like(vl); nv[:-1] = vl[1:]; nv[-1] = lv
            de = rw + gamma*nv*nd - vl
            ac0 = torch.zeros(NE, device='cuda'); ret = torch.zeros_like(vl)
            for t in range(NS-1, -1, -1):
                ac0 = de[t] + gamma*lam*nd[t]*ac0; ret[t] = ac0 + vl[t]
        bo=ob.reshape(-1, obs_dim); ba=ac.reshape(-1); bl=lp.reshape(-1); br=ret.reshape(-1); bv=vl.reshape(-1)
        bad=(ret-vl).reshape(-1); bad=(bad-bad.mean())/(bad.std()+1e-8)
        bs = NE*NS
        for _ in range(ep):
            pr = torch.randperm(bs, device='cuda')
            for st in range(0, bs, mbs):
                idx = pr[st:st+mbs]
                lg, val = pol.forward(bo[idx]); lpa = lg.log_softmax(-1)
                nlp = lpa.gather(-1, ba[idx].unsqueeze(-1)).squeeze(-1)
                ent = -(lpa.exp()*lpa).sum(-1).mean()
                ratio = (nlp - bl[idx]).exp(); adv = bad[idx]
                pg = -torch.min(ratio*adv, ratio.clamp(1-clip, 1+clip)*adv).mean()
                vcl = bv[idx] + (val - bv[idx]).clamp(-clip, clip)
                vloss = 0.5*torch.max((val-br[idx])**2, (vcl-br[idx])**2).mean()
                loss = pg + vc*vloss - ec*ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), mg); opt.step()
    torch.cuda.synchronize(); el = time.time() - t0; sps = tot/el
    print(f"  {tot:,} in {el:.2f}s -> {sps:,.0f} SPS  peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    del env, pol, opt; torch.cuda.empty_cache()
    return sps


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env-only', action='store_true')
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--num-steps', type=int, default=64)
    p.add_argument('--total', type=int, default=10_000_000)
    p.add_argument('--opt2-block', type=int, default=64)
    args = p.parse_args()

    # Force JIT build up-front
    _ = load_opt(); _ = load_opt2()

    env = oracle_mod.CraftaxEnv(1, 0); n_act = env.get_num_actions(); del env
    ne_list = [1024, 4096, 8192, 32768, 65536]
    ro = bench_env_only('oracle', ne_list, n_act)
    torch.cuda.empty_cache()
    r1 = bench_env_only('opt',    ne_list, n_act)
    torch.cuda.empty_cache()
    r2 = bench_env_only('opt2',   ne_list, n_act, block=args.opt2_block)

    print("\n--- env-only comparison (SPS) ---")
    print(f"{'NE':>8} {'oracle':>12} {'opt':>12} {'opt2':>12} {'opt2/oracle':>12}")
    for ne in ne_list:
        print(f"{ne:>8} {ro[ne]:>12,.0f} {r1[ne]:>12,.0f} {r2[ne]:>12,.0f} {r2[ne]/ro[ne]:>11.2f}x")

    if args.env_only:
        raise SystemExit(0)

    to = bench_ppo('oracle', args.num_envs, args.num_steps, args.total)
    t1 = bench_ppo('opt',    args.num_envs, args.num_steps, args.total)
    t2 = bench_ppo('opt2',   args.num_envs, args.num_steps, args.total, block=args.opt2_block)
    print(f"\n--- training SPS ---\n  oracle={to:,.0f}\n  opt   ={t1:,.0f}  ({t1/to:.2f}x)\n  opt2  ={t2:,.0f}  ({t2/to:.2f}x)")
