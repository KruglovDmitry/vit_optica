"""Профили A(ξ) и φ(ξ) подобранного DOE и рельеф на кварце для отчёта.

    python -m vit_optica.plot_doe runs_doe/doe_fit_shared_s4.pt --lens_size 16384 --out shared.png

Без matplotlib пишет только CSV (ξ мм, A, φ рад, рельеф нм по каждой группе и оси).
"""
import argparse
import csv
import math

import torch


def profiles(state, n_coef, L, pix, kind):
    xi = (torch.arange(L) - (L - 1) / 2) * pix                     # как linspace плоскости линзы
    xn = xi / xi.abs().max()
    basis = torch.cos(math.pi * torch.arange(n_coef)[:, None] * (xn[None, :] + 1) / 2)
    out = {}
    for k, p in state.items():
        if 'params.' not in k:
            continue
        g = k.split('params.', 1)[1]
        c = torch.einsum('naqk,kl->naql', p.float().cpu(), basis)[0]   # (ось, A/φ, L)
        for ax, name in ((0, 'x (L1)'), (1, 'y (L3)')):
            A = torch.sigmoid(c[ax, 0]) if kind != 'phase' else torch.ones(L)
            phi = c[ax, 1] if kind != 'amp' else torch.zeros(L)
            out[f'{g} {name}'] = (A, phi - phi.mean())
    return xi * 1e3, out


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a('fit')
    a('--lens_size', type=int, default=16384)
    a('--lens_pix', type=float, default=1.8e-6)
    a('--wavelength', type=float, default=532e-9)
    a('--n', type=float, default=1.46, help='показатель преломления (плавленый кварц)')
    a('--out', default='doe.png')
    args = p.parse_args()
    ck = torch.load(args.fit, map_location='cpu')
    fa = ck.get('args', {})
    xi, prof = profiles(ck['doe'], fa.get('doe_coef', 32), args.lens_size, args.lens_pix,
                        fa.get('doe_kind', 'complex'))
    nm = args.wavelength / (2 * math.pi * (args.n - 1)) * 1e9          # нм рельефа на радиан
    step = max(args.lens_size // 2048, 1)
    with open(args.out.rsplit('.', 1)[0] + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['xi_mm'] + [f'{k} {q}' for k in prof for q in ('A', 'phi_rad', 'relief_nm')])
        for i in range(0, args.lens_size, step):
            w.writerow([f'{xi[i]:.4f}'] + [f'{v:.5f}' for A, ph in prof.values()
                                           for v in (A[i], ph[i], ph[i] * nm)])
    for k, (A, ph) in prof.items():
        print(f'{k}: A {A.min():.2f}…{A.max():.2f}, мощность {(A ** 2).mean():.2f} | '
              f'φ PtV {ph.max() - ph.min():.2f} рад = {(ph.max() - ph.min()) * nm:.0f} нм, '
              f'RMS {ph.std() * nm:.0f} нм')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib нет — записан только CSV')
        return
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for k, (A, ph) in prof.items():
        ax[0].plot(xi[::step], A[::step], label=k)
        ax[1].plot(xi[::step], (ph * nm)[::step], label=k)
    ax[0].set_ylabel('амплитуда A'); ax[1].set_ylabel('рельеф, нм (кварц)')
    ax[1].set_xlabel('ξ, мм'); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    fig.suptitle(f'{args.fit} · L {args.lens_size}')
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
