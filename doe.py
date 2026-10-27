"""Обучаемый DOE в плоскостях линз 4f-схемы.

Скрещенные линзы делят оси: L1 по X строит изображение вектора на столбцы матрицы,
по Y делает Фурье (растяжка); L3 по Y строит изображение строк матрицы на выход,
по X делает Фурье (суммирование). Помехи соседних элементов возникают только на
изображающих осях, поэтому DOE — это два одномерных фильтра:
    hx — в плоскости L1, ось X (вектор -> матрица),
    hy — в плоскости L3, ось Y (строки матрицы -> выход).
h(ξ) = A(ξ)·exp(iφ(ξ)), A = sigmoid(a) ≤ 1; a и φ — суммы косинусов по апертуре
(n_coef коэффициентов), чтобы фильтр был гладким и не подстраивался под сетку.

Оригинальный симулятор не меняется: операторы собираются из тех же PropagatorSinc,
а фазы линз считаются по тем же формулам, что в PropagatorCrossLens.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from vit_optica.model import FastSim

FAMILIES = ('proj', 'qk', 'av', 'ff1', 'ff2')
TERMS = ('pp', 'pn', 'np', 'nn')            # слагаемые split: A±·B±
MODES = ('none', 'fixed', 'shared', 'layer', 'split', 'head')


def _cs(t, dev=None):
    return t.to(dev, torch.complex64) if dev is not None else t.to(torch.complex64)


class DOESim(FastSim):
    """FastSim, в котором изображающие оси линз L1 (X) и L3 (Y) домножены на фильтр.

    X1 = Xa · diag(lens1_x · hx) · Xb,   Y2 = Yd · diag(lens3_y · hy) · Yc.
    При hx = hy = 1 это в точности операторы OpticalMul (проверка check_doe_sim).
    """
    def __init__(self, sim, cfg):
        super().__init__(sim)
        from source.propagator import PropagatorSinc
        dev = sim._propagator_one.operator_X.device
        inp, lens = cfg.input_vector_plane, cfg.first_lens_plane
        mat, out = cfg.matrix_plane, cfg.output_vector_plane
        k, d = cfg.K, cfg.distance
        xl = lens.linspace_by_x.double()
        mx = mat.linspace_by_x.double()
        s1 = PropagatorSinc(inp, lens, cfg)          # вектор -> L1
        s2 = PropagatorSinc(lens, mat, cfg)          # L1 -> матрица
        s3 = PropagatorSinc(mat, lens, cfg)          # матрица -> L3
        s4 = PropagatorSinc(lens, out, cfg)          # L3 -> выход
        # фазы: L1 по X и L3 по Y — изображающие (exp(-ik ξ²/d)); цилиндрическая у матрицы по X
        lens_img = torch.exp(-1j * k / d * xl ** 2)
        cyl = torch.exp(-1j * k / d * mx ** 2)
        self.register_buffer('_xa', _cs(s1.operator_X, dev), persistent=False)              # (Nin_x, L)
        self.register_buffer('_xb', _cs(s2.operator_X * cyl.to(s2.operator_X.dtype)[None, :], dev),
                             persistent=False)                                              # (L, Nmat_x)
        self.register_buffer('_yc', _cs(s3.operator_Y, dev), persistent=False)              # (L, Nmat_y)
        self.register_buffer('_yd', _cs(s4.operator_Y, dev), persistent=False)              # (Nout_y, L)
        self.register_buffer('_lens', _cs(lens_img, dev), persistent=False)                 # (L,)
        del s1, s2, s3, s4
        self._cache = {}
        # для расчётного аподизатора
        self.xi = xl.float()
        self.wavelength, self.distance = cfg.wavelength, d

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _sc(tot, n):
        return slice((tot - n) // 2, (tot - n) // 2 + n)

    def operators(self, gid, hx, hy, vec_shape, mat_shape):
        """X1 (..., Nin, Nmat_x) и Y2 (..., Nout, Nmat_y) для фильтров hx, hy (..., L).
        Кэш по группе и форме: одно и то же в пределах шага используется во всех блоках."""
        key = (gid, tuple(vec_shape), tuple(mat_shape))
        if key in self._cache:
            return self._cache[key]
        xa = self._xa[self._sc(self._xa.shape[0], vec_shape[1]), :]
        xb = self._xb[:, self._sc(self._xb.shape[1], mat_shape[1])]
        yc = self._yc[:, self._sc(self._yc.shape[1], mat_shape[0])]
        yd = self._yd[self._sc(self._yd.shape[0], mat_shape[0]), :]
        lx = self._lens if hx is None else self._lens * hx
        ly = self._lens if hy is None else self._lens * hy
        X1 = (xa * lx[..., None, :]) @ xb
        Y2 = (yd * ly[..., None, :]) @ yc
        self._cache[key] = (X1, Y2)
        return X1, Y2

    def forward(self, A, B, doe=None):
        if doe is None:
            return super().forward(A, B)
        gid, hx, hy = doe
        sim = self.sim
        vec = sim.prepare_vector(A)
        mat = sim.prepare_matrix(B)
        Y1, _ = self._slices(sim._propagator_one, vec.shape[-2:], mat.shape[-2:])
        _, X2 = self._slices(sim._propagator_two, mat.shape[-2:], (mat.size(-2), 1))
        X1, Y2 = self.operators(gid, hx, hy, vec.shape[-2:], mat.shape[-2:])
        y = (Y1 @ vec.new_ones(vec.shape[-2], 1)).squeeze(-1)
        if X1.dim() == 2:
            g = (vec[..., :1, :] @ X1).squeeze(-2)                # (..., M, 2K)
        else:   # свой фильтр на голову: einsum, иначе matmul размножит X1 на батч и строки
            g = torch.einsum('...hmi,hik->...hmk', vec[..., 0, :], X1.squeeze(-3))
        u = g * X2.squeeze(-1)
        h = (u @ mat.squeeze(-3).transpose(-1, -2)) * y
        if Y2.dim() == 2:
            out = h @ Y2.transpose(-1, -2)
        else:
            out = torch.einsum('...hmn,hon->...hmo', h, Y2)
        return sim.prepare_out(out.unsqueeze(-1))


@torch.no_grad()
def check_doe_sim(dsim, device, shapes=((65, 32, 65), (65, 192, 384)), tol=1e-4):
    """DOESim с hx = hy = 1 против FastSim (то же, что OpticalMul)."""
    worst = 0.0
    L = dsim._lens.shape[0]
    one = torch.ones(L, dtype=torch.complex64, device=device)
    for m, k, n in shapes:
        A = torch.rand(2, 1, m, k, device=device)
        B = torch.rand(1, 1, k, n, device=device)
        ref, out = FastSim.forward(dsim, A, B), dsim(A, B, doe=('check', one, one))
        worst = max(worst, ((out - ref).norm() / ref.norm()).item())
    dsim.clear_cache()
    print(f'DOESim: отличие от FastSim при DOE = 1: {worst:.1e} (допуск {tol:.0e})')
    return worst < tol


def nyquist_apodizer(xi, pitch, wavelength, distance, aperture_half):
    """Амплитудный фильтр, при котором ядро помех пиксель→пиксель становится δ
    (1D-модель: прямоугольный элемент шага p, интегрирование по пикселю,
    results/doe_opinion/doe1d.py). Частота в единицах 1/p: f = ξ·p/(λ·d)."""
    f = xi.double() * pitch / (wavelength * distance)
    u = aperture_half * pitch / (wavelength * distance)
    s = torch.zeros_like(f)
    for k in range(-8, 9):
        sh = f + k
        s = s + torch.where(sh.abs() <= u, torch.sinc(sh) ** 2, torch.zeros_like(sh))
    H = torch.where(f.abs() <= u, 1.0 / s.clamp_min(1e-12), torch.zeros_like(f))
    return (H / H.max()).float()


class DOEBank(nn.Module):
    """Параметры фильтров по группам.

    mode: fixed  — расчётный аподизатор (не обучается);
          shared — один DOE на все умножения;
          layer  — свой на семейство (proj, qk, av, ff1, ff2);
          split  — свой на семейство и слагаемое split (pp, pn, np, nn);
          head   — свой на семейство, для qk и av ещё и на каждую голову.
    axes: 'xy', 'x' или 'y' — какие оси фильтруются.
    kind: complex — амплитуда и фаза; phase — только фаза (A = 1, чистый рельеф);
          amp — только амплитуда (φ = 0, серая маска).
    """
    KINDS = ('complex', 'phase', 'amp')

    def __init__(self, dsim, mode, n_coef=32, heads=6, axes='xy', pitch=3.6e-6, kind='complex'):
        super().__init__()
        assert mode in MODES and mode != 'none' and kind in self.KINDS
        self.mode, self.heads, self.axes, self.kind = mode, heads, axes, kind
        self.err = None                                  # погрешности изготовления, set_errors
        self.dsim = [dsim]                               # не регистрируем как подмодуль
        xi = dsim.xi
        xn = xi / xi.abs().max()
        basis = torch.cos(math.pi * torch.arange(n_coef)[:, None] * (xn[None, :] + 1) / 2)
        self.register_buffer('basis', basis, persistent=False)            # (K, L)
        if mode == 'fixed':
            H = nyquist_apodizer(xi, pitch, dsim.wavelength, dsim.distance, xi.abs().max().item())
            self.register_buffer('fixed', H, persistent=False)
            self.params = nn.ParameterDict()
            return
        groups = {}
        for g in self.group_names():
            n = heads if (mode == 'head' and g in ('qk', 'av')) else 1
            p = torch.zeros(n, 2, 2, n_coef)             # (голова, ось x/y, амплитуда/фаза, коэф.)
            p[:, :, 0, 0] = 4.0                          # A ≈ 0.98, φ = 0: почти прозрачный DOE
            groups[g] = nn.Parameter(p)
        self.params = nn.ParameterDict(groups)

    def group_names(self):
        if self.mode == 'shared':
            return ['all']
        if self.mode in ('layer', 'head'):
            return list(FAMILIES)
        return [f'{f}_{t}' for f in FAMILIES for t in TERMS]

    def group(self, fam, term):
        if fam not in FAMILIES:                      # служебные теги (тесты утечки) — как proj
            fam = 'proj'
        if self.mode in ('fixed', 'shared'):
            return 'all'
        if self.mode == 'split':
            return f'{fam}_{term}'
        return fam

    def filters(self, fam, term):
        """(gid, hx, hy); hx, hy — (L,), для голов hx (heads, 1, L), hy (heads, L)."""
        gid = self.group(fam, term)
        if self.mode == 'fixed':
            H = self.fixed.to(torch.complex64)
            hx = H if 'x' in self.axes else None
            hy = H if 'y' in self.axes else None
            return gid, hx, hy
        p = self.params[gid]
        c = torch.einsum('naqk,kl->naql', p, self.basis)          # (n, 2, 2, L)
        A, phi = self._amp_phase(c)
        if self.err is not None:
            A, phi = self._apply_errors(A, phi)
        h = A * torch.exp(1j * phi)                                # (n, 2, L)
        if h.shape[0] == 1:
            hx, hy = h[0, 0], h[0, 1]
        else:   # головы: X1 должен иметь батч (heads, 1) под (B, heads, M), Y2 — (heads,)
            hx, hy = h[:, 0, None, :], h[:, 1]
        return (gid, hx if 'x' in self.axes else None, hy if 'y' in self.axes else None)

    def set_errors(self, shift_mm=(0.0, 0.0), quant=0, amp_err=0.0, corr_mm=1.0, seed=0):
        """Погрешности изготовления и юстировки фиксированной маски (для оценки допусков).
        shift_mm — сдвиг маски поперёк оси (x у L1, y у L3), мм; quant — число уровней
        амплитуды на [0, 1] (0 = без квантования); amp_err — относительная ошибка амплитуды
        (гауссова, сглаженная с корреляцией corr_mm), одна реализация на seed."""
        xi = self.dsim[0].xi
        pix = float(xi[1] - xi[0])
        L = xi.numel()
        idx = []
        for s in shift_mm:
            k = int(round(s * 1e-3 / pix))
            idx.append((torch.arange(L) - k).clamp(0, L - 1))     # край маски продолжается
        noise = None
        if amp_err > 0:
            g = torch.Generator().manual_seed(seed)
            w = torch.randn(2, L, generator=g)
            r = max(int(corr_mm * 1e-3 / pix), 1)
            ker = torch.exp(-0.5 * (torch.arange(-3 * r, 3 * r + 1) / r) ** 2)
            w = torch.nn.functional.conv1d(w[:, None], (ker / ker.norm())[None, None],
                                           padding=3 * r)[:, 0]
            noise = amp_err * w / w.std()
        dev = self.basis.device
        self.err = dict(idx=[i.to(dev) for i in idx], quant=quant,
                        noise=None if noise is None else noise.to(dev))

    def _apply_errors(self, A, phi):
        e = self.err
        A = torch.stack([A[:, a].index_select(-1, e['idx'][a]) for a in range(2)], 1)
        phi = torch.stack([phi[:, a].index_select(-1, e['idx'][a]) for a in range(2)], 1)
        if e['noise'] is not None:
            A = (A * (1 + e['noise'])).clamp(0, 1)
        if e['quant']:
            A = torch.round(A * (e['quant'] - 1)) / (e['quant'] - 1)
        return A, phi

    def _amp_phase(self, c):
        A = torch.sigmoid(c[:, :, 0]) if self.kind != 'phase' else torch.ones_like(c[:, :, 0])
        phi = c[:, :, 1] if self.kind != 'amp' else torch.zeros_like(c[:, :, 1])
        return A, phi

    @torch.no_grad()
    def load_flexible(self, state):
        """Загрузить DOE другого режима: 'proj' -> 'proj_pp' и т.д. (split), одна голова ->
        все головы (head), 'all' -> все группы. Число коэффициентов должно совпадать."""
        src = {k.split('params.', 1)[1]: v for k, v in state.items() if 'params.' in k}
        for g, p in self.params.items():
            fam = g.split('_')[0]
            v = next((src[k] for k in (g, fam, 'all', 'proj') if k in src), None)
            assert v is not None, f'в файле DOE нет группы для {g}: {sorted(src)}'
            if v.shape[0] != p.shape[0]:
                v = v[:1].expand_as(p) if v.shape[0] == 1 else v.mean(0, keepdim=True)
            p.copy_(v)
        return sorted(src)

    def new_step(self):
        self.dsim[0].clear_cache()

    @torch.no_grad()
    def describe(self):
        """Средняя амплитуда и пропускание по мощности для отчёта."""
        if self.mode == 'fixed':
            return {'all': dict(mean_amp=self.fixed.mean().item(),
                                power=(self.fixed ** 2).mean().item())}
        out = {}
        for g, p in self.params.items():
            c = torch.einsum('naqk,kl->naql', p, self.basis)
            A, phi = self._amp_phase(c)
            out[g] = dict(mean_amp=A.mean().item(), power=(A ** 2).mean().item(),
                          phase_rms=phi.std().item())
        return out
