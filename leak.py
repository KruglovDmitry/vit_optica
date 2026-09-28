"""Тесты утечки для ViT.

В ViT внимание двунаправленное, поэтому «утечки будущего», как в языковой
модели, нет: любой патч законно видит все остальные. Проверяются два эффекта,
которые остаются:

1. Перекрёстное влияние соседних позиций (оператор). Меняется один токен j в
   правом операнде и измеряется изменение там, где при точном умножении его
   быть не должно:
     linear — строка j левого операнда: изменение других строк;
     qk     — столбец j у k^T: изменение столбцов l != j (профиль по |l - j|);
     av     — строка j у v при нулевом весе внимания на токен j: любое изменение.
2. Утечка между примерами батча: меняются только остальные примеры батча и
   измеряется изменение результата для примера 0 (оператор и вся модель).
   Нормировка split='global' связывает примеры; shift и split='matrix' — нет.
"""
import torch


def _rel(a, b):
    return (a.norm() / b.norm().clamp_min(1e-30)).item()


@torch.no_grad()
def _op_tests(ctx, T, hd, dim, heads, device, j=None, eps=0.5, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g).to(device)
    j = T // 2 if j is None else j
    res = {}

    # linear: токены — строки левого операнда
    A = rn(2, 1, T, dim)
    W = rn(1, 1, dim, dim) * dim ** -0.5
    A2 = A.clone(); A2[0, 0, j] += eps * rn(dim)
    d = ctx.mm(A2, W, 'leak.lin') - ctx.mm(A, W, 'leak.lin')
    others = torch.cat([d[0, 0, :j], d[0, 0, j + 1:]])
    res['linear_rows'] = _rel(others, d[0, 0, j])
    A3 = A.clone(); A3[1] = rn(1, T, dim)
    res['linear_batch'] = _rel(ctx.mm(A3, W, 'leak.lin')[0] - ctx.mm(A, W, 'leak.lin')[0],
                               ctx.mm(A, W, 'leak.lin')[0])

    # qk: ключи — столбцы k^T
    q, kt = rn(2, heads, T, hd), rn(2, heads, hd, T)
    kt2 = kt.clone(); kt2[0, :, :, j] += eps * rn(heads, hd)
    d = ctx.mm(q, kt2, 'leak.qk') - ctx.mm(q, kt, 'leak.qk')
    dj = d[0, :, :, j]
    mask = torch.ones(T, dtype=torch.bool, device=device); mask[j] = False
    res['qk_cols'] = _rel(d[0][..., mask], dj)
    prof = {}
    for dist in (1, 2, 3, 5, 10):
        cols = [c for c in (j - dist, j + dist) if 0 <= c < T]
        if cols:
            prof[dist] = _rel(d[0][..., cols], dj) / len(cols) ** 0.5
    res['qk_profile'] = prof
    kt3 = kt.clone(); kt3[1] = rn(heads, hd, T)
    base = ctx.mm(q, kt, 'leak.qk')[0]
    res['qk_batch'] = _rel(ctx.mm(q, kt3, 'leak.qk')[0] - base, base)

    # av: вес внимания на токен j равен нулю -> при точном умножении v_j не влияет
    p = torch.softmax(rn(2, heads, T, T), -1)
    p0 = p.clone(); p0[..., j] = 0; p0 = p0 / p0.sum(-1, keepdim=True)
    v = rn(2, heads, T, hd)
    dv = eps * rn(heads, hd)
    v2 = v.clone(); v2[0, :, j] += dv
    d = ctx.mm(p0, v2, 'leak.av') - ctx.mm(p0, v, 'leak.av')
    ref = p[0, :, :, j:j + 1] * dv[:, None, :]      # влияние токена j при обычном весе
    res['av_zero_weight'] = _rel(d[0], ref)
    v3 = v.clone(); v3[1] = rn(heads, T, hd)
    base = ctx.mm(p, v, 'leak.av')[0]
    res['av_batch'] = _rel(ctx.mm(p, v3, 'leak.av')[0] - base, base)
    return res


@torch.no_grad()
def operator_leak_tests(ctx, T, hd, dim, heads, device, seed=0):
    """Возвращает показатели для текущего режима и цифровой контроль."""
    diag = ctx.diag_on
    ctx.diag_on = False
    out = {'optic': _op_tests(ctx, T, hd, dim, heads, device, seed=seed)}
    ctx.force_digital = True
    try:
        out['digital'] = _op_tests(ctx, T, hd, dim, heads, device, seed=seed)
    finally:
        ctx.force_digital = False
        ctx.diag_on = diag
    return out


@torch.no_grad()
def batch_independence_test(model, xa, xb):
    """Логиты примера 0 в батче [x0, xa[1:]] против [x0, xb[1:]] и против x0 одного."""
    model.eval()
    x2 = xb.clone(); x2[0] = xa[0]
    l_a = model(xa)[0]
    l_b = model(x2)[0]
    l_1 = model(xa[:1])[0]
    return dict(other_batch=_rel(l_b - l_a, l_a), single=_rel(l_1 - l_a, l_a),
                argmax_same=bool(l_a.argmax() == l_b.argmax() == l_1.argmax()))
