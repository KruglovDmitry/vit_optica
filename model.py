"""ViT с оптическим матричным умножением.

Формулы восстановления знака совпадают с optic_train (shift / split), добавлен
выбор нормировки split: 'matrix' (по каждой матрице — пример x голова, не зависит
от состава батча) или 'global' (по всему тензору, как в optic_train и в старых
запусках ViT).
"""
import collections
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_OPTICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Optical_matrix_multiplication'))
if _OPTICS_DIR not in sys.path:
    sys.path.insert(1, _OPTICS_DIR)

STUB_SCALE = 3.44e-3   # масштаб выхода реального sim; калибровка его снимает


# ============================================================== симулятор
class StubSim(nn.Module):
    """Заглушка без железа: точное умножение * масштаб.

    noise_sigma — мультипликативный шум (X·Y)∘(1+σξ), модель (6) статьи;
    blur — размытие правого операнда (маски) по обеим осям ядром [b, 1-2b, b]:
    грубая имитация перекрёстного влияния соседних элементов маски, нужна,
    чтобы проверить тесты утечки на CPU.
    """
    def __init__(self, noise_sigma=0.0, blur=0.0):
        super().__init__()
        self.sigma, self.blur = float(noise_sigma), float(blur)

    def _blur(self, Q):
        b = self.blur
        Qp = F.pad(Q, (1, 1, 1, 1))
        Q = (1 - 2 * b) * Qp[..., 1:-1, 1:-1] + b * (Qp[..., 1:-1, :-2] + Qp[..., 1:-1, 2:])
        Qp = F.pad(Q, (0, 0, 1, 1))
        return (1 - 2 * b) * Qp[..., 1:-1, :] + b * (Qp[..., :-2, :] + Qp[..., 2:, :])

    def forward(self, P, Q):
        if self.blur > 0:
            Q = self._blur(Q)
        y = torch.matmul(P, Q) * STUB_SCALE
        if self.sigma > 0:
            y = y * (1.0 + self.sigma * torch.randn_like(y))
        return y


def build_sim(stub, device, aperture=512, lens_size=16384, distance=0.15,
              noise_sigma=0.0, stub_blur=0.0):
    if stub:
        return StubSim(noise_sigma, stub_blur).to(device)
    import source
    cfg = source.Config(
        right_matrix_count_columns=aperture, right_matrix_count_rows=aperture,
        right_matrix_width=3.6e-6 * aperture, right_matrix_height=3.6e-6 * aperture,
        min_height_gap=3.6e-6,
        right_matrix_split_x=2, right_matrix_split_y=2,
        left_matrix_split_x=2, left_matrix_split_y=2, result_matrix_split=2,
        distance=distance, lens_size=lens_size)
    return source.OpticalMul(cfg).to(device)


# ================================================= способы знакового умножения
def mm_shift(ctx, A, B, eps=1e-8):
    k = A.shape[-1]
    sa = torch.clamp(-A.amin(dim=-1, keepdim=True), min=0)     # по строкам A
    sb = torch.clamp(-B.amin(dim=-2, keepdim=True), min=0)     # по столбцам B
    P, Q = A + sa, B + sb
    a = P.amax(dim=-1, keepdim=True).clamp_min(eps)
    b = Q.amax(dim=-2, keepdim=True).clamp_min(eps)
    PQ = ctx.sim(P / a, Q / b) * a * b * ctx.gain_for(A, B)
    return (PQ - sb * A.sum(dim=-1, keepdim=True)
            - sa * B.sum(dim=-2, keepdim=True) - k * sa * sb)


def _split_max(X, how, eps=1e-12):
    if how == 'global':                          # по всему тензору (зависит от батча)
        return X.max().clamp_min(eps)
    return X.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)   # по каждой матрице


def mm_split(ctx, A, B):
    Ap, An = A.clamp(min=0), (-A).clamp(min=0)
    Bp, Bn = B.clamp(min=0), (-B).clamp(min=0)

    def term(X, Y):
        mx, my = _split_max(X, ctx.split_norm), _split_max(Y, ctx.split_norm)
        return ctx.sim(X / mx, Y / my) * mx * my * ctx.gain_for(A, B)

    out = term(Ap, Bp) - term(Ap, Bn)
    if bool((An > 0).any()):                     # attn-веса >= 0: два вызова вместо четырёх
        out = out - term(An, Bp) + term(An, Bn)
    return out


MODES = {'digital': None, 'shift': mm_shift, 'split': mm_split}


class OpticContext:
    """Общее состояние оптики. Не nn.Module: симулятор не попадает в state_dict
    каждого слоя (в старом коде его буферы дублировались в 72 головах)."""
    def __init__(self, mode='digital', sim=None, gain=1.0, split_norm='matrix'):
        assert mode in MODES
        self.mode, self.sim, self.gain, self.split_norm = mode, sim, gain, split_norm
        self.force_digital = False
        self.per_shape_gain = False       # калибровать g отдельно для каждой формы (m, k, n)
        self._gains = {}
        self.diag_on = False
        self.diag = collections.defaultdict(lambda: collections.defaultdict(float))

    @torch.no_grad()
    def gain_for(self, A, B):
        if not self.per_shape_gain:
            return self.gain
        key = (A.shape[-2], A.shape[-1], B.shape[-1])
        if key not in self._gains:
            g = torch.Generator(device='cpu').manual_seed(0)
            P = torch.rand(1, 1, *key[:2], generator=g).to(A.device)
            Q = torch.rand(1, 1, *key[1:], generator=g).to(A.device)
            raw, ref = self.sim(P, Q), P @ Q
            self._gains[key] = ((raw * ref).sum() / (raw * raw).sum()).item()
        return self._gains[key]

    def mm(self, A, B, tag, enabled=True):
        if not enabled or self.mode == 'digital' or self.force_digital:
            return torch.matmul(A, B)
        out = MODES[self.mode](self, A, B)
        if self.diag_on:
            with torch.no_grad():
                ref = torch.matmul(A, B)
                err = out - ref
                d = self.diag[tag]
                d['rel'] += (err.norm() / ref.norm().clamp_min(1e-12)).item()
                d['bias'] += (err.sum() / ref.abs().sum().clamp_min(1e-12)).item()
                d['n'] += 1
        return out

    def diag_clear(self):
        self.diag.clear()

    def diag_summary(self):
        """Относительная ошибка (%) и смещение (%) по семействам умножений."""
        fam = collections.defaultdict(list)
        for tag, d in self.diag.items():
            if d['n']:
                fam[tag.split('.')[-1]].append((d['rel'] / d['n'], d['bias'] / d['n']))
        return {f: dict(rel=100 * sum(r for r, _ in v) / len(v),
                        bias=100 * sum(b for _, b in v) / len(v))
                for f, v in fam.items()}


# ======================================================================= ViT
def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class OpticLinear(nn.Module):
    def __init__(self, fin, fout, ctx, tag):
        super().__init__()
        self.ctx, self.tag, self.optic = ctx, tag, False
        self.weight = nn.Parameter(torch.empty(fin, fout))
        self.bias = nn.Parameter(torch.zeros(fout))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):                          # x: (B, T, fin)
        y = self.ctx.mm(x[:, None], self.weight[None, None], self.tag, self.optic)
        return y[:, 0] + self.bias


class Attention(nn.Module):
    def __init__(self, dim, heads, ctx, idx, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.h, self.hd, self.ctx, self.idx = heads, dim // heads, ctx, idx
        self.q = OpticLinear(dim, dim, ctx, f'b{idx}.proj')
        self.k = OpticLinear(dim, dim, ctx, f'b{idx}.proj')
        self.v = OpticLinear(dim, dim, ctx, f'b{idx}.proj')
        self.o = OpticLinear(dim, dim, ctx, f'b{idx}.proj')
        self.attn_drop, self.proj_drop = nn.Dropout(attn_drop), nn.Dropout(proj_drop)
        self.optic_qk = self.optic_av = False

    def forward(self, x):
        B, T, D = x.shape
        sp = lambda t: t.view(B, T, self.h, self.hd).transpose(1, 2)
        q, k, v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        s = self.ctx.mm(q, k.transpose(-2, -1), f'b{self.idx}.qk', self.optic_qk)
        p = self.attn_drop((s * self.hd ** -0.5).softmax(-1))
        out = self.ctx.mm(p, v, f'b{self.idx}.av', self.optic_av)
        return self.proj_drop(self.o(out.transpose(1, 2).reshape(B, T, D)))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, ctx, idx, drop, drop_path_p, ls_init):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, ctx, idx, attn_drop=drop, proj_drop=drop)
        self.fc1 = OpticLinear(dim, dim * mlp_ratio, ctx, f'b{idx}.ff1')
        self.fc2 = OpticLinear(dim * mlp_ratio, dim, ctx, f'b{idx}.ff2')
        self.drop = nn.Dropout(drop)
        self.g1 = nn.Parameter(ls_init * torch.ones(dim))
        self.g2 = nn.Parameter(ls_init * torch.ones(dim))
        self.dp = drop_path_p

    def forward(self, x):
        x = x + drop_path(self.g1 * self.attn(self.n1(x)), self.dp, self.training)
        h = self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.n2(x))))))
        return x + drop_path(self.g2 * h, self.dp, self.training)


class ConvStem(nn.Module):
    """Свёрточный стебель вместо нарезки на патчи (цифровой, как и патч-эмбеддинг).

    Несколько свёрток 3x3 с шагом 2 (BN + ReLU) уменьшают изображение в 2^n раз,
    затем свёртка с ядром и шагом patch / 2^n доводит его до той же сетки
    (img_size / patch)^2 токенов, что и обычная нарезка. Число токенов и стоимость
    оптики не меняются; добавляется локальность, которой ViT не хватает на малых данных.
    """
    def __init__(self, in_ch, dim, patch, max_stages=4):
        super().__init__()
        n = 0
        while n < max_stages and patch % (2 ** (n + 1)) == 0 and 2 ** (n + 1) <= patch:
            n += 1
        chans = [max(dim // 2 ** (n - i), 16) for i in range(n)]   # напр. 48, 96 при dim=192
        layers, c = [], in_ch
        for co in chans:
            layers += [nn.Conv2d(c, co, 3, 2, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            c = co
        rest = patch // 2 ** n                       # 28 -> 7, 20 -> 5, степени двойки -> 1
        layers.append(nn.Conv2d(c, dim, rest, rest))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ViT(nn.Module):
    """token_order='random' — фиксированная перестановка патчей. Для цифровой
    модели ничего не меняет (эмбеддинги позиций обучаемые), но делает соседей
    по индексу несоседними в изображении: так проверяется, важна ли структура
    перекрёстного влияния соседних позиций."""
    def __init__(self, ctx, img_size=64, patch=8, in_ch=3, num_classes=200,
                 dim=192, depth=12, heads=6, mlp_ratio=2, drop=0.1,
                 drop_path=0.1, ls_init=1e-4, token_order='raster', order_seed=0,
                 stem='patch'):
        super().__init__()
        assert img_size % patch == 0 and dim % heads == 0
        self.ctx = ctx
        self.n_patches = (img_size // patch) ** 2
        assert stem in ('patch', 'conv')
        self.patch = (nn.Conv2d(in_ch, dim, patch, patch) if stem == 'patch'
                      else ConvStem(in_ch, dim, patch))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        if token_order == 'random':
            g = torch.Generator().manual_seed(order_seed)
            perm = torch.randperm(self.n_patches, generator=g)
        else:
            perm = torch.arange(self.n_patches)
        self.register_buffer('perm', perm, persistent=True)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_ratio, ctx, i, drop, dpr[i], ls_init) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    @property
    def seq_len(self):
        return self.n_patches + 1

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)[:, self.perm]
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x)[:, 0])


# ============================================================ локализация
def resolve_optic_where(where):
    alias = {'all': 'ff+proj+qk+av', 'none': '', 'attn': 'qk+av'}
    parts = [p for p in alias.get(where, where).split('+') if p]
    bad = set(parts) - {'ff', 'proj', 'qk', 'av'}
    if bad:
        raise ValueError(f'неизвестные компоненты оптики: {sorted(bad)}')
    return {c: c in parts for c in ('ff', 'proj', 'qk', 'av')}


def optic_block_indices(depth, n):
    """n блоков с оптикой, равномерно по глубине (0 = все)."""
    if n <= 0 or n >= depth:
        return list(range(depth))
    return sorted({round(i * (depth - 1) / (n - 1)) for i in range(n)}) if n > 1 else [0]


def apply_optic_where(model, where, optic_layers=0):
    f = resolve_optic_where(where)
    on = set(optic_block_indices(len(model.blocks), optic_layers))
    for i, b in enumerate(model.blocks):
        use = i in on
        b.fc1.optic = b.fc2.optic = use and f['ff']
        for l in (b.attn.q, b.attn.k, b.attn.v, b.attn.o):
            l.optic = use and f['proj']
        b.attn.optic_qk = use and f['qk']
        b.attn.optic_av = use and f['av']
    tag = '+'.join(c for c in f if f[c]) or 'none'
    blocks = sorted(on) if tag != 'none' else []
    print(f'ОПТИКА: [{tag}] в блоках {blocks}')
    return f, blocks
