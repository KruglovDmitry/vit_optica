"""Сводка JSON-результатов в CSV и markdown-таблицу.

    python -m vit_optica.collect_results ./runs_vit            # печать
    python -m vit_optica.collect_results ./runs_vit out.csv    # + CSV
"""
import csv
import json
import sys
from pathlib import Path

COLS = ['name', 'mode', 'where', 'blocks', 'lens', 'split_norm', 'order', 'stem', 'phase',
        'epochs', 'val_acc', 'test_acc', 'test_top5', 'test_acc_best_val',
        'err_ff1', 'err_ff2', 'err_proj', 'err_qk', 'err_av',
        'leak_qk_cols', 'leak_av_zero', 'leak_batch_model', 'wall_min']


def row(p):
    d = json.loads(p.read_text())
    a, le, op = d['args'], d.get('layer_errors', {}), d.get('leak', {})
    o = op.get('operator', {}).get('optic', {})
    tb = d.get('test_at_best_val') or {}
    f = lambda x: '' if x is None else (f'{x:.2f}' if isinstance(x, float) else x)
    g = lambda x: '' if x is None else f'{x:.1e}'
    phase = 'infer' if d['inference'] else ('finetune' if d['load_ckpt'] else 'train')
    return dict(
        name=p.stem, mode=d['mode'], where=d['optic_where'] if d['mode'] != 'digital' else 'none',
        blocks=len(d.get('optic_blocks', [])), lens=d['lens_size'], split_norm=d['split_norm'],
        order=d['token_order'], stem=a.get('stem', 'patch') + (f"x{a['stem_width']}" if a.get('stem') == 'conv' and a.get('stem_width', 1.0) != 1.0 else ''), phase=phase, epochs=0 if d['inference'] else a['epochs'],
        val_acc=f(d['val']['acc']), test_acc=f(d['test']['acc']), test_top5=f(d['test']['top5']),
        test_acc_best_val=f(tb.get('acc')),
        **{f'err_{k}': f(le.get(k, {}).get('rel')) for k in ('ff1', 'ff2', 'proj', 'qk', 'av')},
        leak_qk_cols=g(o.get('qk_cols')), leak_av_zero=g(o.get('av_zero_weight')),
        leak_batch_model=g(op.get('model_batch', {}).get('other_batch')),
        wall_min=f(d['wall_sec'] / 60))


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else './runs_vit')
    probes = sorted(src.glob('probe_*.json'))
    if probes:
        print('Замеры скорости (прогноз на весь бюджет):')
        for p in probes:
            d = json.loads(p.read_text())
            print(f"  {p.stem[6:]:40s} {d['sec_per_iter']:.3f} с/итер → "
                  f"{d['projected_hours']:.1f} ч ({d['projected_hours'] / 24:.1f} сут)")
        print()
    rows = sorted((row(p) for p in src.glob('*.json') if not p.name.startswith('probe_')),
                  key=lambda r: (r['phase'], r['mode'], r['where'], str(r['lens'])))
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(rows)
    print('| ' + ' | '.join(COLS) + ' |')
    print('|' + '---|' * len(COLS))
    for r in rows:
        print('| ' + ' | '.join(str(r[c]) for c in COLS) + ' |')


if __name__ == '__main__':
    main()