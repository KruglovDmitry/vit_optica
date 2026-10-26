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

from vit_optica.model import FastSim, check_fast_sim
from vit_optica.doe import DOEBank, DOESim, check_doe_sim, MODES as DOE_MODES
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
    a('--stem', default='patch', choices=['patch', 'conv'],
      help='patch — нарезка на патчи; conv — свёрточный стебель (цифровой, те же токены)')
    a('--stem_width', type=float, default=1.0,
      help='множитель каналов стебля: 1.0 -> 48/96, 0.5 -> 24/48 (при h_dim 192, патч 28)')
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
    a('--fast_sim', type=int, default=1,
      help='1: тот же расчёт симулятора с переставленным порядком умножений (быстрее и '
           'экономнее по памяти); при старте сверяется с исходным, при расхождении — откат')
    a('--doe', default='none', choices=DOE_MODES,
      help='фильтр в плоскостях линз (vit_optica/doe.py): fixed — расчётный аподизатор; '
           'shared / layer / split / head — обучаемый, общий / на семейство / на слагаемое '
           'split / на голову (qk, av)')
    a('--doe_axes', default='xy', choices=['xy', 'x', 'y'])
    a('--doe_init', default='', help='начальный DOE из файла doe_fit.py (или чекпоинта)')
    a('--doe_freeze', type=int, default=0, help='1: DOE не обучается (только веса сети)')
    a('--doe_coef', type=int, default=32, help='косинусов на амплитуду и на фазу')
    a('--doe_lr', type=float, default=1e-2)
    a('--doe_refresh', type=int, default=50,
      help='каждые N шагов пересчитывать калибровку gain под текущий DOE')
    a('--sim_split', type=int, default=2,
      help='подотсчётов сетки на элемент 3,6 мкм; для --doe нужно >= 4 (см. build_sim)')
    a('--lens_pix', type=float, default=1.8e-6, help='шаг сетки линзы; апертура = lens_size·lens_pix')
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
    a('--mixup_alpha', type=float, default=0.0,
      help='>0: MixUp на батчах, где не применён CutMix')
    a('--eval_every', type=int, default=5)
    a('--max_eval_batches', type=int, default=0, help='0 = вся выборка')
    a('--eval_only', type=int, default=0)
    a('--probe_iters', type=int, default=0,
      help='замер скорости: N итераций обучения, прогноз времени на --epochs, выход')
    a('--load_ckpt', default='')
    a('--save_ckpt', type=int, default=1)
    a('--ckpt_every', type=int, default=5,
      help='каждые N эпох сохранять состояние обучения для продолжения (0 = не сохранять)')
    a('--resume', type=int, default=1,
      help='1: если есть last_<имя>.pt в out_dir, продолжить обучение с него')
    a('--seed', type=int, default=1337)
    # диагностика
    a('--leak_tests', type=int, default=1)
    a('--out_dir', default='./runs_vit')
    a('--tensorboard', type=int, default=1, help='писать кривые в <out_dir>/tb/<имя запуска>')
    a('--comment', default='')
    return p.parse_args()


def mixup(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


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
def evaluate(model, loader, device, max_batches=0, progress=''):
    """progress — подпись для печати хода оценки (для долгих оптических прогонов)."""
    model.eval()
    n = correct = correct5 = 0
    loss = 0.0
    total = min(len(loader), max_batches) if max_batches else len(loader)
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if progress and i and (i % 10 == 0):
            el = time.time() - t0
            print(f'  {progress}: {i}/{total} батчей, {el / 60:.1f} мин, '
                  f'осталось ≈ {el / i * (total - i) / 60:.1f} мин', flush=True)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        top5 = logits.topk(min(5, logits.shape[-1]), -1).indices
        correct += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(-1).sum().item()
        n += y.numel()
    model.train()
    return dict(loss=loss / n, acc=100 * correct / n, top5=100 * correct5 / n)


def probe(args, model, opt, tl, device, lr_at, name_hint, warmup=3, ctx_doe_step=None):
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
        if ctx_doe_step is not None:
            ctx_doe_step(i)
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
    if device == 'cuda':
        print(f'Устройство: cuda:{torch.cuda.current_device()} '
              f'({torch.cuda.get_device_name()}), видно карт: {torch.cuda.device_count()}')
    else:
        print('ВНИМАНИЕ: CUDA недоступна, обучение идёт на CPU')
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
        use_doe = args.doe != 'none'
        if use_doe:
            assert args.mode == 'split' and not args.stub_sim and not args.sim_parallel, \
                '--doe работает с --mode split, реальным sim и без --sim_parallel'
            assert args.calibrate == 2, '--doe требует --calibrate 2 (gain на группу DOE)'
        sim = build_sim(args.stub_sim, device, args.aperture, args.lens_size,
                        args.distance, args.noise_sigma, args.stub_blur,
                        fast=bool(args.fast_sim) and not args.stub_sim, doe=use_doe,
                        split=args.sim_split, lens_pix=args.lens_pix)
        if use_doe:
            assert check_doe_sim(sim, device, shapes=((T, hd, T), (T, args.h_dim, args.h_dim * args.mlp_ratio))), \
                'DOESim при DOE = 1 расходится с FastSim'
        elif isinstance(sim, FastSim) and not check_fast_sim(
                sim, device, shapes=((T, hd, T), (T, args.h_dim, args.h_dim * args.mlp_ratio))):
            print('ВНИМАНИЕ: FastSim расходится с исходным расчётом — используется исходный')
            sim = sim.sim
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
    ck = None
    if args.load_ckpt:
        ck = torch.load(args.load_ckpt, map_location=device)
        ck_cfg = ck.get('config', {})
        for key, default in (('token_order', 'raster'), ('stem', 'patch'), ('stem_width', 1.0)):
            val = ck_cfg.get(key, default)
            if val != getattr(args, key):
                print(f'ВНИМАНИЕ: --{key} берётся из чекпоинта ({val}), а не {getattr(args, key)}')
                setattr(args, key, val)
    model = ViT(ctx, img_size, patch, 3, n_cls, args.h_dim, args.depth, args.heads,
                args.mlp_ratio, args.drop, args.drop_path, args.ls_init,
                args.token_order, stem=args.stem, stem_width=args.stem_width).to(device)
    bank = None
    if args.mode != 'digital' and args.doe != 'none':
        bank = DOEBank(sim, args.doe, args.doe_coef, args.heads, args.doe_axes).to(device)
        model.doe = bank                     # параметры DOE попадают в state_dict и в оптимизатор
        ctx.doe = bank
        if args.doe_init:
            st = torch.load(args.doe_init, map_location=device)
            st = st.get('doe', st)
            src = bank.load_flexible(st.get('state_dict', st))
            print(f'DOE загружен из {args.doe_init} (группы в файле: {src})')
        if args.doe_freeze:
            bank.requires_grad_(False)
        print(f'DOE: {args.doe}, оси {args.doe_axes}, групп {max(len(bank.params), 1)}, '
              f'параметров {sum(p.numel() for p in bank.parameters())}')
    if ck is not None:
        miss, extra = model.load_state_dict(ck['state_dict'], strict=False)
        bad = [k for k in miss if not k.startswith('doe.')] + list(extra)
        assert not bad, f'чекпоинт не совпадает с моделью: {bad[:5]}'
        if any(k.startswith('doe.') for k in ck['state_dict']):
            print('DOE взят из чекпоинта')
        print(f'загружены веса {args.load_ckpt} (режим обучения: {ck.get("mode")})')
    flags, blocks = apply_optic_where(model, args.optic_where if args.mode != 'digital'
                                      else 'none', args.optic_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Параметров: {n_params / 1e6:.2f}M, токенов {T}, размерность головы {hd}')

    # ---------------------------------------------------------------- обучение
    epochs = 0 if args.eval_only else args.epochs
    doe_params = [p for p in bank.parameters() if p.requires_grad] if bank is not None else []
    doe_ids = {id(p) for p in doe_params}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in doe_ids], lr_scale=1.0)]
    if doe_params:
        groups.append(dict(params=doe_params, weight_decay=0.0, lr_scale=args.doe_lr / args.lr))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(len(tl) // args.grad_accum, 1)
    total, warm = epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        t = (s - warm) / max(total - warm, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    def doe_step(s):
        """Новый шаг: сбросить кэш операторов DOE; раз в doe_refresh шагов — калибровку gain."""
        if bank is None or bank.mode == 'fixed' or args.doe_freeze:
            return
        bank.new_step()
        if s % args.doe_refresh == 0:
            ctx._gains.clear()

    def doe_eval():
        if bank is not None and bank.mode != 'fixed':
            bank.new_step(); ctx._gains.clear()     # DOE мог измениться — пересчитать всё

    if args.probe_iters:
        probe(args, model, opt, tl, device, lr_at, name_hint=f"{args.mode}_{args.comment or 'run'}",
              ctx_doe_step=doe_step)
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
    t0, step, start_ep, wall_before = time.time(), 0, 0, 0.0
    last_path = Path(args.out_dir) / f'last_{run_name}.pt'
    if epochs and args.resume and last_path.exists():
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        step, start_ep, log, best = st['step'], st['epoch'], st['log'], st['best']
        wall_before = st['wall']
        torch.set_rng_state(st['rng_cpu'])
        if torch.cuda.is_available() and st.get('rng_cuda') is not None:
            torch.cuda.set_rng_state(st['rng_cuda'])
        print(f'ПРОДОЛЖЕНИЕ с эпохи {start_ep + 1} из {last_path}')
    model.train()
    for ep in range(start_ep, epochs):
        te0 = time.time()
        tot_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            doe_step(step)
            use_cm = args.cutmix_prob > 0 and torch.rand(1).item() < args.cutmix_prob
            if use_cm or args.mixup_alpha > 0:
                x, ya, yb, lam = cutmix(x, y) if use_cm else mixup(x, y, args.mixup_alpha)
                logits = model(x)                           # один прямой проход
                loss = (lam * F.cross_entropy(logits, ya, label_smoothing=args.label_smoothing)
                        + (1 - lam) * F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing))
            else:
                loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
            (loss / args.grad_accum).backward()
            tot_loss += loss.item(); nb += 1
            if (i + 1) % args.grad_accum == 0:
                for gp in opt.param_groups:
                    gp['lr'] = lr_at(step) * gp.get('lr_scale', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
        rec = dict(epoch=ep + 1, train_loss=tot_loss / max(nb, 1), lr=lr_at(step),
                   epoch_sec=time.time() - te0)
        if (ep + 1) % args.eval_every == 0 or ep + 1 == epochs:
            doe_eval()
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
        if args.ckpt_every and ((ep + 1) % args.ckpt_every == 0) and ep + 1 < epochs:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            tmp = last_path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                            epoch=ep + 1, log=log, best=best,
                            wall=wall_before + time.time() - t0,
                            rng_cpu=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                            config=vars(args)), tmp)
            tmp.replace(last_path)                      # атомарно: не испортить при обрыве
    wall = wall_before + time.time() - t0
    if epochs and last_path.exists():
        last_path.unlink()                              # обучение завершено — продолжать нечего

    # ---------------------------------------------------------------- итоговая оценка
    pr = lambda name: name if args.mode != 'digital' else ''
    doe_eval()
    res_val = evaluate(model, vl, device, args.max_eval_batches, pr('val'))
    res_test = evaluate(model, te, device, args.max_eval_batches, pr('test'))
    res_test_best = None
    if best['state'] is not None and best['epoch'] != epochs:
        last = copy.deepcopy(model.state_dict())
        model.load_state_dict(best['state'])
        doe_eval()
        res_test_best = dict(evaluate(model, te, device, args.max_eval_batches),
                             epoch=best['epoch'], val_acc=best['acc'])
        model.load_state_dict(last)
        doe_eval()

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
    gval = lambda v: v if isinstance(v, float) else v.flatten().tolist()
    res = dict(gains_per_shape={str(k): gval(v) for k, v in ctx._gains.items()},
               doe=dict(mode=args.doe, axes=args.doe_axes, coef=args.doe_coef,
                        lr=args.doe_lr, filters=bank.describe()) if bank is not None else None,
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