"""Допуски фиксированной маски: ошибка умножения при погрешностях изготовления и юстировки.

Берёт DOE из doe_fit.py и считает относительную ошибку по семействам (как doe_fit, со
знаковой калибровкой) при сдвиге маски, квантовании амплитуды, ошибке профиля и шуме
детектора. Точность сети для выбранных точек — через main.py --eval_only с теми же флагами.

    python -m vit_optica.doe_tol runs_doe/doe_fit_shared_amp_s4_L12288.pt --lens_size 12288
"""
import argparse
import json

import torch

from vit_optica.model import build_sim, OpticContext
from vit_optica.doe import DOEBank
from vit_optica.doe_fit import shapes, sample


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a('fit')
    a('--lens_size', type=int, default=16384)
    a('--distance', type=float, default=0.15)
    a('--sim_split', type=int, default=4)
    a('--shifts', default='0,0.05,0.1,0.2,0.5,1', help='мм, сдвиг обеих масок')
    a('--quants', default='0,4,8,16,32', help='уровней амплитуды')
    a('--amp_errs', default='0,0.01,0.02,0.05', help='относительная ошибка амплитуды')
    a('--noises', default='0,0.001,0.003,0.01', help='шум детектора, доля полной шкалы')
    a('--seeds', type=int, default=3, help='реализаций для ошибки профиля')
    a('--out', default='')
    args = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.fit, map_location=dev)
    fa = ck['args']
    sim = build_sim(False, dev, fa['aperture'], args.lens_size, args.distance, doe=True,
                    split=args.sim_split, lens_pix=fa['lens_pix'])
    ctx = OpticContext('split', sim, 1.0)
    ctx.per_shape_gain = True
    ctx.calib_signed = True
    bank = DOEBank(sim, fa['doe'], fa['doe_coef'], fa['heads'], fa['doe_axes'],
                   kind=fa.get('doe_kind', 'complex')).to(dev)
    bank.load_flexible(ck['doe'])
    bank.requires_grad_(False)
    ctx.doe = bank
    hd = fa['h_dim'] // fa['heads']
    shp = shapes(fa['tokens'], fa['h_dim'], hd, fa['mlp_ratio'])
    torch.manual_seed(123)
    fixed = {f: sample(shp[f], shp[f][2], 4, fa['heads'] if f in ('qk', 'av') else 1, dev)
             for f in shp}

    @torch.no_grad()
    def errs():
        bank.new_step(); ctx._gains.clear()
        torch.manual_seed(7)
        out = {}
        for f, (A, B) in fixed.items():
            r = A @ B
            out[f] = round(100 * ((ctx.mm(A, B, f'b0.{f}') - r).norm() / r.norm()).item(), 2)
        out['mean'] = round(sum(out.values()) / len(out), 2)
        return out

    res = {}

    def run(name, **kw):
        noise = kw.pop('noise', 0.0)
        seeds = kw.pop('seeds', 1)
        acc = []
        for s in range(seeds):
            bank.err = None
            if kw:
                bank.set_errors(seed=s, **kw)
            ctx.det_noise = noise
            acc.append(errs())
        e = {k: round(sum(x[k] for x in acc) / len(acc), 2) for k in acc[0]}
        res[name] = e
        print(f'{name:28s} ' + ' '.join(f'{k} {v:6.2f}' for k, v in e.items()), flush=True)

    run('номинал')
    ctx.doe = None; run('без DOE'); ctx.doe = bank
    fl = lambda s: [float(x) for x in s.split(',') if x]
    for s in fl(args.shifts)[1:]:
        run(f'сдвиг {s} мм (обе)', shift_mm=(s, s))
        run(f'сдвиг {s} мм (только L1)', shift_mm=(s, 0.0))
    for q in fl(args.quants)[1:]:
        run(f'уровней {int(q)}', quant=int(q))
    for e in fl(args.amp_errs)[1:]:
        run(f'ошибка амплитуды {e:.0%}', amp_err=e, seeds=args.seeds)
    for n in fl(args.noises)[1:]:
        run(f'шум детектора {n:.1%}', noise=n)
        ctx.doe = None; run(f'шум {n:.1%} без DOE', noise=n); ctx.doe = bank
    ctx.det_noise = 0.0
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'→ {args.out}')


if __name__ == '__main__':
    main()
