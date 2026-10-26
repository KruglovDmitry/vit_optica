"""Подбор DOE без сети: минимизируем ошибку оптического умножения на случайных матрицах.

Формы умножений — как в ViT из runs_c100 (T токенов, ширина h, голова hd, MLP×r).
Для каждой группы DOE (см. DOEBank) берётся своя форма; ошибка — относительная L2
после лучшего общего множителя (его в сети снимает калибровка gain).

    python -m vit_optica.doe_fit --doe layer --sim_split 8 --lens_size 16384 \
        --distance 0.15 --steps 800 --out doe_layer_s8.pt

Результат грузится в main.py через --doe_init (и --doe_freeze 1 для инференса).
"""
import argparse
import json
import time

import torch

from vit_optica.model import build_sim, OpticContext
from vit_optica.doe import DOEBank, check_doe_sim, FAMILIES


def shapes(T, h, hd, r):
    """(A, B) формы и знаковость: av — веса внимания >= 0."""
    return {'proj': ((T, h), (h, h), True), 'qk': ((T, hd), (hd, T), True),
            'av': ((T, T), (T, hd), False), 'ff1': ((T, h), (h, h * r), True),
            'ff2': ((T, h * r), (h * r, h), True)}


def sample(shp, signed, batch, heads, device):
    (m, k), (k2, n) = shp[0], shp[1]
    A = torch.randn(batch, heads, m, k, device=device)
    if not signed:
        A = torch.softmax(A * 2, -1)                  # как строки внимания
    B = torch.randn(1 if heads == 1 else batch, heads, k2, n, device=device)
    return A, B


def rel_err(y, r, fit_scale):
    if fit_scale:
        y = y * (y * r).sum() / (y * y).sum()
    return ((y - r).norm() / r.norm())


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a('--doe', default='layer', choices=['shared', 'layer', 'split', 'head'])
    a('--doe_axes', default='xy')
    a('--doe_coef', type=int, default=32)
    a('--aperture', type=int, default=512)
    a('--lens_size', type=int, default=16384)
    a('--lens_pix', type=float, default=1.8e-6)
    a('--distance', type=float, default=0.15)
    a('--sim_split', type=int, default=8)
    a('--tokens', type=int, default=65)
    a('--h_dim', type=int, default=192)
    a('--heads', type=int, default=6)
    a('--mlp_ratio', type=int, default=2)
    a('--steps', type=int, default=800)
    a('--batch', type=int, default=4)
    a('--lr', type=float, default=2e-2)
    a('--seed', type=int, default=0)
    a('--out', default='doe_fit.pt')
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    hd = args.h_dim // args.heads
    sim = build_sim(False, dev, args.aperture, args.lens_size, args.distance, doe=True,
                    split=args.sim_split, lens_pix=args.lens_pix)
    assert check_doe_sim(sim, dev, shapes=((args.tokens, hd, args.tokens),))
    ctx = OpticContext('split', sim, 1.0)
    ctx.per_shape_gain = True
    bank = DOEBank(sim, args.doe, args.doe_coef, args.heads, args.doe_axes).to(dev)
    ctx.doe = bank
    opt = torch.optim.Adam(bank.parameters(), lr=args.lr)
    shp = shapes(args.tokens, args.h_dim, hd, args.mlp_ratio)
    fams = ['proj'] if args.doe == 'shared' else list(FAMILIES)

    def batch_for(f):
        heads = args.heads if f in ('qk', 'av') else 1
        return sample(shp[f], shp[f][2], args.batch, heads, dev)

    def run(fit_scale, A_B=None):
        out = {}
        for f in fams:
            A, B = A_B[f] if A_B else batch_for(f)
            y = ctx.mm(A, B, f'b0.{f}')
            out[f] = rel_err(y, A @ B, fit_scale)
        return out

    @torch.no_grad()
    def evaluate(fixed):
        bank.new_step(); ctx._gains.clear()
        return {f: 100 * v.item() for f, v in run(False, fixed).items()}

    torch.manual_seed(123)
    fixed = {f: batch_for(f) for f in fams}
    torch.manual_seed(args.seed)
    ctx_doe, ctx.doe = ctx.doe, None
    base = evaluate(fixed)                            # без DOE
    ctx.doe = ctx_doe
    print('без DOE, %:', {k: round(v, 2) for k, v in base.items()}, flush=True)
    t0, log = time.time(), []
    for it in range(args.steps):
        bank.new_step(); ctx._gains.clear()
        errs = run(True)
        loss = sum(e ** 2 for e in errs.values()) / len(errs)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (it + 1) % max(args.steps // 10, 1) == 0 or it + 1 == args.steps:
            ev = evaluate(fixed)
            log.append(dict(step=it + 1, train=100 * loss.sqrt().item(), eval=ev))
            print(f'шаг {it + 1}: train {100 * loss.sqrt().item():.2f}% | eval ' +
                  ', '.join(f'{k} {v:.2f}' for k, v in ev.items()) +
                  f' | {time.time() - t0:.0f} с', flush=True)
    desc = bank.describe()
    torch.save(dict(doe=bank.state_dict(), args=vars(args), base=base, final=log[-1]['eval'],
                    describe=desc), args.out)
    print('DOE:', json.dumps(desc, ensure_ascii=False))
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
