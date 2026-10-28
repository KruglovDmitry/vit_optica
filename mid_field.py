"""Поле в плоскости матрицы (середина 4f-схемы) для входного вектора из единиц.

По X линза L1 строит изображение вектора (здесь работает маска hx), по Y — Фурье
(растяжка вектора на строки матрицы; hx на эту ось не действует). Поле — произведение
y(η)·g(ξ), картинка интенсивности |y|²·|g|². Считается без DOE и с маской из doe_fit.py.

    python -m vit_optica.mid_field runs_doe/doe_fit_shared_amp_s4_L12288.pt --lens_size 12288 \
        --k 192 --out mid_L12288.png
"""
import argparse

import numpy as np
import torch

from vit_optica.model import build_sim
from vit_optica.doe import DOEBank


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a('fit')
    a('--lens_size', type=int, default=16384)
    a('--distance', type=float, default=0.15)
    a('--sim_split', type=int, default=4)
    a('--k', type=int, default=192, help='длина вектора из единиц (элементов 3,6 мкм)')
    a('--fam', default='proj', help='семейство, чей фильтр брать (для layer)')
    a('--out', default='mid_field.png')
    args = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.fit, map_location=dev)
    fa = ck['args']
    ds = build_sim(False, dev, fa['aperture'], args.lens_size, args.distance, doe=True,
                   split=args.sim_split, lens_pix=fa['lens_pix'])
    bank = DOEBank(ds, fa['doe'], fa['doe_coef'], fa['heads'], fa['doe_axes'],
                   kind=fa.get('doe_kind', 'complex')).to(dev)
    bank.load_flexible(ck['doe'])
    with torch.no_grad():
        _, hx, _ = bank.filters(args.fam, 'pp')
        sim, cfg = ds.sim, ds.cfg
        vec = sim.prepare_vector(torch.ones(1, 1, 1, args.k, device=dev))   # (1,1,1,sy,Nx)
        row = vec[0, 0, 0, 0]
        sc = lambda tot, n: slice((tot - n) // 2, (tot - n) // 2 + n)
        xa = ds._xa[sc(ds._xa.shape[0], row.numel()), :]
        f = row @ xa                                            # поле в плоскости L1 по X
        g0 = (f * ds._lens) @ ds._xb
        g1 = (f * ds._lens * hx) @ ds._xb
        Y = sim._propagator_one.operator_Y
        y = Y[:, sc(Y.shape[-1], vec.shape[-2])] @ torch.ones(vec.shape[-2], 1, dtype=Y.dtype, device=dev)
        y = y.squeeze(-1)
        xm = cfg.matrix_plane.linspace_by_x.cpu().numpy() * 1e6
        ym = cfg.matrix_plane.linspace_by_y.cpu().numpy() * 1e6
        xl = ds.xi.cpu().numpy() * 1e3
        I0, I1 = (g0.abs() ** 2).cpu().numpy(), (g1.abs() ** 2).cpu().numpy()
        Iy = (y.abs() ** 2).cpu().numpy()
        Fl = (f.abs() ** 2).cpu().numpy()
        H = hx.abs().cpu().numpy()
    half = args.k * 3.6 / 2
    inside = np.abs(xm) < half - 3.6 * 2                         # без двух крайних элементов

    def stats(I, ref):
        m = I[inside].mean()
        rip = (I[inside].max() - I[inside].min()) / m
        out = I[np.abs(xm) > half + 3.6 * 2].max() / ref
        return m / ref, rip, out
    ref = I0[inside].mean()
    for name, I in (('без DOE', I0), ('с маской', I1)):
        m, rip, out = stats(I, ref)
        print(f'{name:9s}: средний уровень {m:.3f} (без DOE = 1), пульсации внутри {100 * rip:.1f}% PtV, '
              f'засветка за краем вектора до {100 * out:.1f}%')
    np.savez(args.out.rsplit('.', 1)[0] + '.npz', x_um=xm, y_um=ym, I_x_nodoe=I0, I_x_doe=I1,
             I_y=Iy, xi_lens_mm=xl, I_lens=Fl, mask_amp=H)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib нет — записан только npz'); return
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(xm, I0 / ref, label='без DOE'); ax[0, 0].plot(xm, I1 / ref, label='с маской')
    for e in (-half, half):
        ax[0, 0].axvline(e, color='k', lw=.5, ls='--')
    ax[0, 0].set_xlim(-half * 1.25, half * 1.25)
    ax[0, 0].set_title(f'X (изображение, L1+маска): вектор из {args.k} единиц'); ax[0, 0].set_xlabel('x, мкм')
    ax[0, 0].legend()
    ax[0, 1].plot(ym, Iy / Iy.max()); ax[0, 1].set_title('Y (Фурье, растяжка; маска hx не действует)')
    ax[0, 1].set_xlabel('y, мкм')
    ax[1, 0].plot(xl, Fl / Fl.max(), label='|поле|² на L1 по X'); ax[1, 0].plot(xl, H, label='амплитуда маски')
    ax[1, 0].set_xlabel('ξ, мм'); ax[1, 0].set_title('плоскость L1'); ax[1, 0].legend()
    img = np.outer(Iy / Iy.max(), I1 / ref)
    sel = np.abs(xm) < half * 1.25
    ax[1, 1].imshow(img[:, sel], aspect='auto', cmap='inferno',
                    extent=[xm[sel][0], xm[sel][-1], ym[-1], ym[0]])
    ax[1, 1].set_title('плоскость матрицы, с маской'); ax[1, 1].set_xlabel('x, мкм'); ax[1, 1].set_ylabel('y, мкм')
    fig.suptitle(f'{args.fit} · L {args.lens_size}')
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
