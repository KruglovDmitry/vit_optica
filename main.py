"""ViT с оптическим умножением: обучение, инференс на готовых весах, тесты утечки.

Режимы как в optic_train:
  digital — torch.matmul; shift — сдвиг (1 вызов sim); split — разложение (до 4 вызовов).

Примеры (рабочая папка — родитель пакета):
  python -m vit_optica.main --dataset tinyimagenet --mode digital --epochs 200 --comment digi
  python -m vit_optica.main --dataset tinyimagenet --mode split --optic_where all \
      --eval_only 1 --load_ckpt runs_vit/ckpt_digital_digi.pt --lens_size 16384 --comment inf
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vit_optica import (build_sim, OpticContext, ViT, apply_optic_where,
                        ParallelSim, check_parallel, build_datasets, build_loaders,
                        DATASET_DEFAULTS, operator_leak_tests, batch_independence_test)


def get_args():
    p = argparse.ArgumentParser()
    a = p.add_argument
    # данные
    a('--dataset', default='tinyimagenet', choices=list(DATASET_DEFAULTS))
    a('--data_dir', default='./data')
    a('--download', type=int, default=0)
    a('--img_size', type=int, default=0, help='0 = по умолчанию для датасета')
    a('--patch', type=int, default=0, help='0 = по умолчанию для датасета')
    a('--val_fraction', type=float, default=-1, help='-1 = по умолчанию для датасета')
    a('--autoaug', type=int, default=1)
    a('--workers', type=int, default=8)
    # модель
    a('--h_dim', type=int, default=192)
    a('--depth', type=int, default=12)
    a('--heads', type=int, default=6)
    a('--mlp_ratio', type=int, default=2)
    a('--drop', type=float, default=0.1)
    a('--drop_path', type=float, default=0.1)
    a('--ls_init', type=float, default=1e-4)
    a('--token_order', default='raster', choices=['raster', 'random'])
    # оптика
    a('--mode', default='digital', choices=['digital', 'shift', 'split'])
    a('--optic_where', default='all', help="ff, proj, qk, av через '+'; all / attn / none")
    a('--optic_layers', type=int, default=0, help='число блоков с оптикой (0 = все)')
    a('--split_norm', default='matrix', choices=['matrix', 'global'])
    a('--aperture', type=int, default=512)
    a('--lens_size', type=int, default=16384)
    a('--distance', type=float, default=0.15)
    a('--calibrate', type=int, default=2,
      help='0 = gain из --gain; 1 = один gain по матрицам aperture x aperture; '
           '2 = отдельный gain для каждой формы умножения')
    a('--gain', type=float, default=1.0)
    a('--stub_sim', action='store_true')
    a('--stub_blur', type=float, default=0.0)
    a('--noise_sigma', type=float, default=0.0)
    a('--sim_parallel', type=int, default=0)
    a('--sim_devices', default='')
    a('--check_parallel', type=int, default=1)
    # обучение
    a('--epochs', type=int, default=200)
    a('--warmup_epochs', type=int, default=10)
    a('--batch_size', type=int, default=128)
    a('--grad_accum', type=int, default=1)
    a('--eval_batch_size', type=int, default=128)
    a('--lr', type=float, default=1e-3)
    a('--min_lr', type=float, default=1e-5)
    a('--wd', type=float, default=0.05)
    a('--label_smoothing', type=float, default=0.1)
    a('--cutmix_prob', type=float, default=0.5)
    a('--eval_every', type=int, default=5)
    a('--max_eval_batches', type=int, default=0, help='0 = вся выборка')
    a('--eval_only', type=int, default=0)
    a('--probe_iters', type=int, default=0,
      help='замер скорости: N итераций обучения, прогноз времени на --epochs, выход')
    a('--load_ckpt', default='')
    a('--save_ckpt', type=int, default=1)
    a('--seed', type=int, default=1337)
    # диагностика
    a('--leak_tests', type=int, default=1)
    a('--out_dir', default='./runs_vit')
    a('--tensorboard', type=int, default=1, help='писать кривые в <out_dir>/tb/<имя запуска>')
    a('--comment', default='')
    return p.parse_args()


def cutmix(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[-2:]
    r = math.sqrt(1 - lam)
    ch, cw = int(H * r), int(W * r)
    cy, cx = torch.randint(H, (1,)).item(), torch.randint(W, (1,)).item()
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[idx], lam


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    n = correct = correct5 = 0
    loss = 0.0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        top5 = logits.topk(min(5, logits.shape[-1]), -1).indices
        correct += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(-1).sum().item()
        n += y.numel()
    model.train()
    return dict(loss=loss / n, acc=100 * correct / n, top5=100 * correct5 / n)


def probe(args, model, opt, tl, device, lr_at, name_hint, warmup=3):
    """Время итерации обучения (прямой + обратный проход) и прогноз на весь бюджет."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    model.train()
    it = iter(tl)
    times = []
    for i in range(warmup + args.probe_iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        sync(); t = time.time()
        loss = F.cross_entropy(model(x), y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        sync()
        if i >= warmup:
            times.append(time.time() - t)
    s_it = sorted(times)[len(times) // 2]
    hours = s_it * len(tl) * args.epochs / 3600
    print(f'ЗАМЕР {name_hint}: {s_it:.3f} с/итерацию (батч {args.batch_size}), '
          f'{len(tl)} итераций/эпоху → {args.epochs} эпох ≈ {hours:.1f} ч ({hours / 24:.1f} сут)')
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f'probe_{name_hint}.json').write_text(json.dumps(dict(
        args=vars(args), sec_per_iter=s_it, iters_per_epoch=len(tl),
        projected_hours=hours), ensure_ascii=False, indent=2))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dcfg = DATASET_DEFAULTS[args.dataset]
    patch = args.patch or dcfg['patch']

    # ---------------------------------------------------------------- данные
    train, val, test, n_cls, img_size = build_datasets(
        args.dataset, args.data_dir, args.val_fraction, split_seed=0,
        download=bool(args.download), autoaug=bool(args.autoaug),
        img_size=args.img_size or None)
    tl, vl, te = build_loaders(train, val, test, args.batch_size, args.eval_batch_size,
                               args.workers, args.seed)
    print(f'Данные: {args.dataset}, {img_size}x{img_size}, патч {patch}, классов {n_cls}; '
          f'train {len(train)}, val {len(val)}, test {len(test)}')

    # ---------------------------------------------------------------- оптика
    T = (img_size // patch) ** 2 + 1
    hd = args.h_dim // args.heads
    widest = max(args.h_dim, args.h_dim * args.mlp_ratio, T, hd)
    if args.mode != 'digital':
        assert widest <= args.aperture, (
            f'узкая ось {widest} > апертуры {args.aperture}: уменьши mlp_ratio/h_dim/'
            f'число токенов или увеличь --aperture')
    sim = None
    if args.mode != 'digital':
        sim = build_sim(args.stub_sim, device, args.aperture, args.lens_size,
                        args.distance, args.noise_sigma, args.stub_blur)
        if args.sim_parallel and torch.cuda.device_count() > 1:
            devs = ([int(d) for d in args.sim_devices.split(',') if d.strip()]
                    or list(range(torch.cuda.device_count())))
            sim = ParallelSim(sim, devs, output_device=devs[0])
            print(f'SIM-ШАРДИНГ по строкам левого операнда на карты {devs}')
            if args.check_parallel:
                check_parallel(sim, args.aperture, hd, T, device)
    gain = args.gain
    if sim is not None and args.calibrate:
        with torch.no_grad():
            P = torch.rand(1, 1, args.aperture, args.aperture, device=device)
            Q = torch.rand(1, 1, args.aperture, args.aperture, device=device)
            raw, ref = sim(P, Q), P @ Q
            gain = ((raw * ref).sum() / (raw * raw).sum()).item()
        print(f'Калибровка gain: {gain:.4f}')
    ctx = OpticContext(args.mode, sim, gain, args.split_norm)
    ctx.per_shape_gain = sim is not None and args.calibrate == 2

    # ---------------------------------------------------------------- модель
    model = ViT(ctx, img_size, patch, 3, n_cls, args.h_dim, args.depth, args.heads,
                args.mlp_ratio, args.drop, args.drop_path, args.ls_init,
                args.token_order).to(device)
    if args.load_ckpt:
        ck = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(ck['state_dict'], strict=True)
        print(f'загружены веса {args.load_ckpt} (режим обучения: {ck.get("mode")})')
        ck_order = ck.get('config', {}).get('token_order', 'raster')
        if ck_order != args.token_order:
            print(f'ВНИМАНИЕ: порядок токенов берётся из чекпоинта ({ck_order}), '
                  f'а не из --token_order {args.token_order}')
            args.token_order = ck_order
    flags, blocks = apply_optic_where(model, args.optic_where if args.mode != 'digital'
                                      else 'none', args.optic_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Параметров: {n_params / 1e6:.2f}M, токенов {T}, размерность головы {hd}')

    # ---------------------------------------------------------------- обучение
    epochs = 0 if args.eval_only else args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(len(tl) // args.grad_accum, 1)
    total, warm = epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        t = (s - warm) / max(total - warm, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    if args.probe_iters:
        probe(args, model, opt, tl, device, lr_at, name_hint=f"{args.mode}_{args.comment or 'run'}")
        return

    run_name = f"{args.mode}_{args.comment or 'run'}"
    tb = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(Path(args.out_dir) / 'tb' / run_name)
            tb.add_text('config', json.dumps(vars(args), ensure_ascii=False, indent=1))
        except ImportError:
            print('tensorboard не установлен (pip install tensorboard) — кривые не пишутся')

    log, best = [], dict(acc=-1.0, epoch=-1, state=None)
    t0, step = time.time(), 0
    model.train()
    for ep in range(epochs):
        te0 = time.time()
        tot_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.cutmix_prob > 0 and torch.rand(1).item() < args.cutmix_prob:
                x, ya, yb, lam = cutmix(x, y)
                logits = model(x)                           # один прямой проход
                loss = (lam * F.cross_entropy(logits, ya, label_smoothing=args.label_smoothing)
                        + (1 - lam) * F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing))
            else:
                loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
            (loss / args.grad_accum).backward()
            tot_loss += loss.item(); nb += 1
            if (i + 1) % args.grad_accum == 0:
                for gp in opt.param_groups:
                    gp['lr'] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
        rec = dict(epoch=ep + 1, train_loss=tot_loss / max(nb, 1), lr=lr_at(step),
                   epoch_sec=time.time() - te0)
        if (ep + 1) % args.eval_every == 0 or ep + 1 == epochs:
            v = evaluate(model, vl, device, args.max_eval_batches)
            rec.update(val_acc=v['acc'], val_loss=v['loss'])
            if v['acc'] > best['acc']:
                best = dict(acc=v['acc'], epoch=ep + 1,
                            state=copy.deepcopy(model.state_dict()))
        log.append(rec)
        if tb is not None:
            tb.add_scalar('Loss/train', rec['train_loss'], ep + 1)
            tb.add_scalar('LR', rec['lr'], ep + 1)
            tb.add_scalar('Time/epoch_sec', rec['epoch_sec'], ep + 1)
            if 'val_acc' in rec:
                tb.add_scalar('Accuracy/val', rec['val_acc'], ep + 1)
                tb.add_scalar('Loss/val', rec['val_loss'], ep + 1)
        print(f"эпоха {ep + 1:4d} | loss {rec['train_loss']:.3f} | lr {rec['lr']:.2e} | "
              f"val {rec.get('val_acc', float('nan')):5.2f}% | {rec['epoch_sec']:.0f} с")
    wall = time.time() - t0

    # ---------------------------------------------------------------- итоговая оценка
    res_val = evaluate(model, vl, device, args.max_eval_batches)
    res_test = evaluate(model, te, device, args.max_eval_batches)
    res_test_best = None
    if best['state'] is not None and best['epoch'] != epochs:
        last = copy.deepcopy(model.state_dict())
        model.load_state_dict(best['state'])
        res_test_best = dict(evaluate(model, te, device, args.max_eval_batches),
                             epoch=best['epoch'], val_acc=best['acc'])
        model.load_state_dict(last)

    # послойная ошибка на одном валидационном батче
    layer_err = {}
    if args.mode != 'digital':
        ctx.diag_clear(); ctx.diag_on = True
        with torch.no_grad():
            model.eval(); model(next(iter(vl))[0].to(device)); model.train()
        ctx.diag_on = False
        layer_err = ctx.diag_summary()
        print('ОШИБКА ПО СЕМЕЙСТВАМ (отн. %, смещение %): ' +
              ', '.join(f"{k}={v['rel']:.2f}/{v['bias']:+.2f}" for k, v in layer_err.items()))

    # тесты утечки
    leak = {}
    if args.leak_tests and args.mode != 'digital':
        leak['operator'] = operator_leak_tests(ctx, T, hd, args.h_dim, args.heads, device)
        batches = []
        for loader in (vl, te):
            for x, _ in loader:
                if len(x) >= 2:
                    batches.append(x.to(device))
                if len(batches) == 2:
                    break
            if len(batches) == 2:
                break
        xa, xb = batches
        n = min(len(xa), len(xb), 16)
        leak['model_batch'] = batch_independence_test(model, xa[:n], xb[:n])
        o = leak['operator']['optic']
        print(f"УТЕЧКА (оператор): linear строки {o['linear_rows']:.1e} | qk столбцы "
              f"{o['qk_cols']:.1e} | av при нулевом весе {o['av_zero_weight']:.1e} | "
              f"между примерами: lin {o['linear_batch']:.1e} qk {o['qk_batch']:.1e} "
              f"av {o['av_batch']:.1e}")
        mb = leak['model_batch']
        print(f"УТЕЧКА (модель, между примерами): другой батч {mb['other_batch']:.1e}, "
              f"один пример {mb['single']:.1e}, argmax совпадает: {mb['argmax_same']}")

    print('=' * 60)
    where = args.optic_where if args.mode != 'digital' else 'none'
    print(f"РЕЖИМ {args.mode} [{where}] lens {args.lens_size} | "
          f"val {res_val['acc']:.2f}% | test {res_test['acc']:.2f}% "
          f"(top-5 {res_test['top5']:.2f}%) | {wall / 60:.1f} мин")
    print('=' * 60)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = f"{args.mode}_{args.comment or 'run'}"
    if args.save_ckpt and epochs > 0:
        torch.save(dict(mode=args.mode, state_dict=model.state_dict(),
                        best_state_dict=best['state'], config=vars(args),
                        gain=gain, test=res_test), out / f'ckpt_{name}.pt')
    res = dict(gains_per_shape={str(k): v for k, v in ctx._gains.items()},
               args=vars(args), dataset=args.dataset, mode=args.mode,
               optic_where=args.optic_where, optic_blocks=blocks,
               lens_size=args.lens_size, split_norm=args.split_norm,
               token_order=args.token_order, inference=bool(args.eval_only),
               load_ckpt=args.load_ckpt, n_params=n_params, tokens=T, gain=gain,
               val=res_val, test=res_test, test_at_best_val=res_test_best,
               layer_errors=layer_err, leak=leak, wall_sec=wall, log=log)
    (out / f'{name}.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
    if tb is not None:
        step = max(epochs, 1)
        tb.add_scalar('Final/val_acc', res_val['acc'], step)
        tb.add_scalar('Final/test_acc', res_test['acc'], step)
        tb.add_scalar('Final/test_top5', res_test['top5'], step)
        for fam, v in layer_err.items():
            tb.add_scalar(f'LayerError/{fam}_rel_pct', v['rel'], step)
        tb.close()
    print(f'→ {out / (name + ".json")}')


if __name__ == '__main__':
    main()