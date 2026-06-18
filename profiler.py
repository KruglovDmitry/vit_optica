"""
outlier_profiler.py

Лёгкий профайлер локализации выбросов для блока внимания ViT.

Идея (см. разбор формулы optics_matmul):
  пер-строчная / пер-столбцовая нормировка СЛЕПА по контракционной оси.
  Поэтому для каждого операнда измеряем, насколько выбросы сконцентрированы
  именно на оси свёртки (там, где формула их НЕ локализует):

    QK^T   : свёртка по head_dim  -> опасны КАНАЛЫ  (Q,K:  contraction = -1)
    attn@V : свёртка по seq_k     -> опасны ТОКЕНЫ  (attn: -1,  V: -2)

Метрики на операнд (по «строкам» вдоль контракции):
  dyn_median  -- медиана отношения max/median вдоль контракции.
                 Высокий => почти каждая строка имеет большой динамический
                 диапазон, т.е. её max задаёт устойчивый выброс на оси свёртки.
  dyn_p99     -- тот же показатель, хвост (редкие пиковые строки).
  top1_share  -- доля случаев, когда ОДИН и тот же контракционный индекс
                 оказывается пиком. Высокий => выброс «прибит» к одному
                 каналу/токену (классический channel / attention-sink outlier).
  argmax_H    -- нормированная энтропия распределения argmax (0..1).
                 Низкая => концентрация на немногих индексах.

Решающее правило:
  dyn_median высокий  И  (top1_share высокий / argmax_H низкий)
    => выброс на слепой (контракционной) оси
    => текущей shift_fg недостаточно, нужна эквилибрация вдоль свёртки.
  иначе -- текущая формула покрывает, эквилибрация не нужна.
"""
import math
from collections import defaultdict

import torch


@torch.no_grad()
def _axis_stats(x, contraction_dim):
    """Возвращает (dyn_median, dyn_p99, freq) для одного тензора."""
    x = x.detach().float().abs().movedim(contraction_dim, -1)   # [..., K]
    K = x.shape[-1]
    x = x.reshape(-1, K)                                         # [N, K]

    row_max = x.amax(dim=-1)
    row_med = x.median(dim=-1).values.clamp_min(1e-12)
    dyn = row_max / row_med                                      # диапазон каждой строки

    arg = x.argmax(dim=-1)                                       # какой контр-индекс = пик
    freq = torch.bincount(arg, minlength=K).float()
    return dyn.median().item(), dyn.quantile(0.99).item(), freq.cpu()


class OutlierProfiler:
    def __init__(self):
        self.enabled = False
        self._freq = {}                       # key -> накопленный freq-тензор
        self._dyn_med = defaultdict(float)
        self._dyn_p99 = defaultdict(float)
        self._count = defaultdict(int)

    def reset(self):
        self._freq.clear()
        self._dyn_med.clear()
        self._dyn_p99.clear()
        self._count.clear()

    @torch.no_grad()
    def observe(self, layer_idx, head_idx, name, x, contraction_dim):
        # Полный no-op, когда профилирование выключено (нулевая цена при обучении).
        if not self.enabled:
            return
        med, p99, freq = _axis_stats(x, contraction_dim)
        key = (layer_idx, head_idx, name)
        self._freq[key] = freq if key not in self._freq else self._freq[key] + freq
        self._dyn_med[key] += med
        self._dyn_p99[key] += p99
        self._count[key] += 1

    def _per_head(self):
        rows = []
        for key, c in self._count.items():
            layer, head, name = key
            freq = self._freq[key]
            total = freq.sum().clamp_min(1)
            p = freq / total
            nz = p[p > 0]
            H = float(-(nz * nz.log()).sum() / math.log(len(p))) if len(p) > 1 else 0.0
            rows.append(dict(
                layer=layer, head=head, name=name,
                dyn_median=self._dyn_med[key] / c,
                dyn_p99=self._dyn_p99[key] / c,
                top1_share=float(freq.max() / total),
                argmax_H=H,
            ))
        return rows

    def report(self, dangerous_dyn=8.0, dangerous_top1=0.25):
        """Печатает агрегат по (layer, op), усреднённый по головам."""
        rows = self._per_head()
        if not rows:
            print("OutlierProfiler: нет данных (был ли enabled=True во время прохода?)")
            return

        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for m in ("dyn_median", "dyn_p99", "top1_share", "argmax_H"):
                agg[(r["layer"], r["name"])][m].append(r[m])

        order = {"Q": 0, "K": 1, "V": 2, "attn": 3}
        print(f"{'layer':>5} {'op':>5} {'dyn_med':>8} {'dyn_p99':>9} "
              f"{'top1':>6} {'H':>5}  flag")
        for (layer, name) in sorted(agg, key=lambda kv: (kv[0], order.get(kv[1], 9))):
            d = agg[(layer, name)]
            dm = sum(d["dyn_median"]) / len(d["dyn_median"])
            dp = sum(d["dyn_p99"]) / len(d["dyn_p99"])
            t1 = sum(d["top1_share"]) / len(d["top1_share"])
            H = sum(d["argmax_H"]) / len(d["argmax_H"])
            # attn -- доброкачественный операнд ([0,1]), не помечаем как опасный
            danger = (name != "attn" and dm >= dangerous_dyn and t1 >= dangerous_top1)
            flag = "<-- выброс на контр-оси" if danger else ""
            print(f"{layer:>5} {name:>5} {dm:>8.1f} {dp:>9.1f} "
                  f"{t1:>6.2f} {H:>5.2f}  {flag}")


# Глобальный синглтон -- импортируется в model.py, чтобы не тащить объект
# через все конструкторы. Включается только на время профилирующего прохода.
PROFILER = OutlierProfiler()