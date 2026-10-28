#!/bin/bash
# Обучаемый DOE в плоскостях линз (vit_optica/doe.py), CIFAR-100, ViT как в runs_c100.
# Рабочая точка: L = 16384, d = 0,15 (L/d = 109k, NA 0,098), split.
# Сетка симулятора split 4 (δ = 0,9 мкм): на сетке 2 DOE подстраивается под её артефакт.
#
#   CK=/path/ckpt_digital_patch_s1337.pt DATA=/path/data GPU=0 bash vit_optica/run_doe.sh probe
#   фазы: probe | fit | d0 | ctrl | d1 | d2 | d3 | d4 | d5_frozen | d5_train | d5_nodoe | d5_clear | d5_digital
#   другая линза: L=8192 bash run_doe.sh fit, затем L=8192 bash run_doe.sh d0
set -e
export PYTHONUNBUFFERED=1   # строки эпох сразу в лог (иначе буфер stdout копится до конца)
CK=${CK:?"задай CK — цифровой чекпоинт digital_patch_s1337"}
DATA=${DATA:?"задай DATA — папка с CIFAR-100"}
GPU=${GPU:-0}
OUT=${OUT:-./runs_doe}
SPLIT=${SPLIT:-4}
L=${L:-16384}
FT_EPOCHS=${FT_EPOCHS:-10}
PY=${PY:-python}
NPROC=${NPROC:-1}   # >1: DDP на NPROC карт из GPU (эффективный батч BS·ACC·NPROC)
EXTRA=${EXTRA:-}   # доп. аргументы main.py (ставятся последними)
# обучение через оптику со всеми 12 блоками при батче 128 не влезает в 80 ГБ:
# батч 64 с накоплением 2 (эффективно те же 128)
BS=${BS:-64}
ACC=${ACC:-2}
mkdir -p "$OUT"
SEED=${SEED:-1337}
ST=$([[ "$SEED" == 1337 ]] && echo "" || echo "_s$SEED")   # метка сида
LT=$([[ "$L" == 16384 ]] && echo "" || echo "_L$L")    # метка линзы в именах файлов
DOE_MODE=${DOE_MODE:-layer}   # режим DOE для fit и d0 (layer | shared)
DOE_KIND=${DOE_KIND:-complex}  # complex | phase (только рельеф) | amp (только амплитуда)
KT=$([[ "$DOE_KIND" == complex ]] && echo "" || echo "_$DOE_KIND")
FIT=${FIT:-$OUT/doe_fit_${DOE_MODE}${KT}_s$SPLIT$LT.pt}
EPOCHS=${EPOCHS:-100}       # обучение с нуля (D5)
SCR="--epochs $EPOCHS --warmup_epochs 10 --lr 1e-3 --eval_every 5"

COMMON="--dataset cifar100 --data_dir $DATA --out_dir $OUT --seed $SEED \
  --h_dim 192 --depth 12 --heads 6 --mlp_ratio 2 --aperture 512 --distance 0.15 \
  --lens_size $L --mode split --optic_where all --calibrate 2 --sim_split $SPLIT \
  --batch_size $BS --grad_accum $ACC --eval_batch_size 128 --leak_tests 0"
CS=${CS:-1}   # знаковая калибровка в дообучениях
FT="--calib_signed $CS --load_ckpt $CK --epochs $FT_EPOCHS --warmup_epochs 1 --lr 1e-4 --min_lr 1e-6 --eval_every 1"

run() {   # run <comment> <args...>
  local c="$1"; shift
  echo ">>> [$(date +%H:%M:%S)] $c"
  local launcher="$PY"
  if [[ "$NPROC" -gt 1 ]]; then   # DDP: GPU — список карт через запятую, батч BS — на одну карту
    launcher="$PY -m torch.distributed.run --standalone --nproc_per_node $NPROC"
  fi
  CUDA_VISIBLE_DEVICES=$GPU $launcher -m vit_optica.main $COMMON "$@" $EXTRA --comment "$c" 2>&1 | tee "$OUT/log_$c.txt"
}

case "${1:-}" in
  probe)   # скорость итерации с DOE и без, прогноз времени дообучения
    run probe_nodoe $FT --probe_iters 20
    run probe_doe_layer $FT --doe layer --probe_iters 20 ;;
  fit)     # D0, шаг 1: DOE на ошибку умножения, без сети (минуты)
    CUDA_VISIBLE_DEVICES=$GPU $PY -m vit_optica.doe_fit --doe $DOE_MODE --doe_kind $DOE_KIND --sim_split $SPLIT \
      --lens_size $L --distance 0.15 --steps 800 --out "$FIT" \
      2>&1 | tee "$OUT/log_doe_fit_$DOE_MODE$KT$LT.txt" ;;
  d0)      # D0, шаг 2: инференс цифровых весов без DOE и с подобранным DOE (та же сетка),
           # с прежней калибровкой gain и со знаковой (--calib_signed 1)
    for CS in ${D0_CS:-0 1}; do                        # D0_CS=1 — только знаковая калибровка
      [[ "${D0_NODOE:-1}" == 1 ]] && run d0_nodoe_cs$CS$LT --eval_only 1 --load_ckpt "$CK" --calib_signed $CS
      run d0_fit_${DOE_MODE}${KT}_cs$CS$LT --eval_only 1 --load_ckpt "$CK" --doe $DOE_MODE \
        --doe_kind $DOE_KIND --calib_signed $CS \
        --doe_init "$FIT" --doe_freeze 1
    done ;;
  ctrl)    # контроль: дообучение без DOE
    run ft_nodoe $FT ;;
  d1)      run ft_doe_shared $FT --doe shared --doe_init "$FIT" ;;
  d2)      run ft_doe_layer  $FT --doe layer  --doe_init "$FIT" ;;
  d3)      run ft_doe_split  $FT --doe split  --doe_init "$FIT" ;;
  d4)      run ft_doe_head   $FT --doe head   --doe_init "$FIT" ;;
  # D5: с нуля через оптику; DOE из подбора — заморожен или учится вместе с сетью
  d5_frozen) MT=$([[ "$DOE_MODE" == layer ]] && echo "" || echo "_$DOE_MODE")
    run scr_doe_frozen$MT$KT$LT$ST $SCR --calib_signed $CS --doe $DOE_MODE --doe_kind $DOE_KIND \
      --doe_init "$FIT" --doe_freeze 1 ;;
  d5_train)  run scr_doe_train$LT$ST $SCR --calib_signed $CS --doe layer --doe_init "$FIT" --doe_lr 1e-3 ;;
  d5_nodoe)  run scr_nodoe$LT$ST $SCR --calib_signed $CS ;;   # контроль: оптика без DOE
  d5_clear)  run scr_clear$LT$ST $SCR --calib_signed $CS --doe layer --doe_freeze 1 ;;   # тот же контроль через DOESim: прозрачный DOE (A≈0,98, φ=0), быстрее
  d5_digital)   # цифровой эталон на тех же эпохах
    run scr_digital_e$EPOCHS$ST $SCR --mode digital ;;
  *) echo "usage: CK=... DATA=... bash run_doe.sh [probe|fit|d0|ctrl|d1|d2|d3|d4|d5_frozen|d5_train|d5_nodoe|d5_clear|d5_digital]"; exit 1 ;;
esac
