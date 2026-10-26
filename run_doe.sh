#!/bin/bash
# Обучаемый DOE в плоскостях линз (vit_optica/doe.py), CIFAR-100, ViT как в runs_c100.
# Рабочая точка: L = 16384, d = 0,15 (L/d = 109k, NA 0,098), split.
# Сетка симулятора split 4 (δ = 0,9 мкм): на сетке 2 DOE подстраивается под её артефакт.
#
#   CK=/path/ckpt_digital_patch_s1337.pt DATA=/path/data GPU=0 bash vit_optica/run_doe.sh probe
#   фазы: probe | fit | d0 | ctrl | d1 | d2 | d3 | d4
set -e
CK=${CK:?"задай CK — цифровой чекпоинт digital_patch_s1337"}
DATA=${DATA:?"задай DATA — папка с CIFAR-100"}
GPU=${GPU:-0}
OUT=${OUT:-./runs_doe}
SPLIT=${SPLIT:-4}
L=${L:-16384}
FT_EPOCHS=${FT_EPOCHS:-10}
PY=${PY:-python}
mkdir -p "$OUT"

COMMON="--dataset cifar100 --data_dir $DATA --out_dir $OUT --seed 1337 \
  --h_dim 192 --depth 12 --heads 6 --mlp_ratio 2 --aperture 512 --distance 0.15 \
  --lens_size $L --mode split --optic_where all --calibrate 2 --sim_split $SPLIT \
  --batch_size 128 --eval_batch_size 128 --leak_tests 0"
FT="--load_ckpt $CK --epochs $FT_EPOCHS --warmup_epochs 1 --lr 1e-4 --min_lr 1e-6 --eval_every 1"

run() {   # run <comment> <args...>
  local c="$1"; shift
  echo ">>> [$(date +%H:%M:%S)] $c"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m vit_optica.main $COMMON "$@" --comment "$c" 2>&1 | tee "$OUT/log_$c.txt"
}

case "${1:-}" in
  probe)   # скорость итерации с DOE и без, прогноз времени дообучения
    run probe_nodoe $FT --probe_iters 20
    run probe_doe_layer $FT --doe layer --probe_iters 20 ;;
  fit)     # D0, шаг 1: DOE на ошибку умножения, без сети (минуты)
    CUDA_VISIBLE_DEVICES=$GPU $PY -m vit_optica.doe_fit --doe layer --sim_split $SPLIT \
      --lens_size $L --distance 0.15 --steps 800 --out "$OUT/doe_fit_layer_s$SPLIT.pt" \
      2>&1 | tee "$OUT/log_doe_fit.txt" ;;
  d0)      # D0, шаг 2: инференс цифровых весов без DOE и с подобранным DOE (та же сетка)
    run d0_nodoe --eval_only 1 --load_ckpt "$CK"
    run d0_fit --eval_only 1 --load_ckpt "$CK" --doe layer \
      --doe_init "$OUT/doe_fit_layer_s$SPLIT.pt" --doe_freeze 1 ;;
  ctrl)    # контроль: дообучение без DOE
    run ft_nodoe $FT ;;
  d1)      run ft_doe_shared $FT --doe shared --doe_init "$OUT/doe_fit_layer_s$SPLIT.pt" ;;
  d2)      run ft_doe_layer  $FT --doe layer  --doe_init "$OUT/doe_fit_layer_s$SPLIT.pt" ;;
  d3)      run ft_doe_split  $FT --doe split  --doe_init "$OUT/doe_fit_layer_s$SPLIT.pt" ;;
  d4)      run ft_doe_head   $FT --doe head   --doe_init "$OUT/doe_fit_layer_s$SPLIT.pt" ;;
  *) echo "usage: CK=... DATA=... bash run_doe.sh [probe|fit|d0|ctrl|d1|d2|d3|d4]"; exit 1 ;;
esac
