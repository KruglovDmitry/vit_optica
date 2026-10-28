#!/bin/bash
# Некогерентная схема с реальными устройствами, геометрия стенда (results/devices, 28.10.2026):
# ЖК Sharp LS055R1SX04 (пиксель 47,25 мкм, зелёный субпиксель), элемент 3×3 пикселя = 141,75 мкм,
# поле 32×32 (большие матрицы ViT режутся на блоки), d = 0,1 м, ирис NA 0,02 (апертура 4 мм).
# Инференс digital_patch_s1337 на CIFAR-100, как run_c100_inc.sh.
#   DATA_ARGS — аргументы датасета, как в stage1; RUNS — какие прогоны (по умолчанию все);
#   GPU — номер карты; EXTRA — добавочные аргументы, TAG — суффикс имени прогона. Пример: GPU=1 RUNS="S0 S4" DATA_ARGS="..." bash run_stand.sh
set -e
DATA_ARGS=${DATA_ARGS:?"задай DATA_ARGS как в stage1"}
CK=${CK:-/wd/runs_c100/ckpt_digital_patch_s1337.pt}
OUT=${OUT:-/wd/runs_c100_stand}
PY=${PY:-/wd/venv/bin/python}
RUNS=${RUNS:-"S0 S1 S2 S3 S4 S5"}
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

COMMON="$DATA_ARGS --mode split --optic_where all --calibrate 2 --eval_only 1 --load_ckpt $CK \
  --physics incoherent --out_dir $OUT --leak_tests 0 \
  --inc_field 32 --inc_pitch 141.75e-6 --distance 0.1 --lens_size 1000 --inc_lens_pix 4e-6 \
  --inc_src_fill 0.3333 --inc_mask_fill 0.3333 --inc_mask_fill_y 0.9 --inc_wavelength 525e-9"
OPT="--inc_spectrum 35e-9,5 --inc_glass N-BK7"
LCD="--inc_src_levels 256 --inc_src_gamma 2.2 --inc_src_contrast 300 --inc_src_nonuni 0.02 \
     --inc_mask_levels 256 --inc_mask_gamma 2.2 --inc_mask_contrast 300 --inc_mask_nonuni 0.02"
CAM="--inc_full_well 4e5 --inc_read_noise 50 --inc_background 0.01"
DIM="--inc_full_well 4e4 --inc_read_noise 50 --inc_background 0.01"

declare -A ARGS=(
  [S0]=""                       # идеальные устройства, геометрия стенда
  [S1]="$OPT"                   # + спектр LED 35 нм и одиночные линзы N-BK7
  [S2]="$LCD"                   # + ЖК: 8 бит, гамма 2,2, контраст 300:1, разброс 2 %
  [S3]="$CAM"                   # + камера: 4·10⁵ e⁻ на детектор, шум 50 e⁻, рассеянный свет 1 %
  [S4]="$OPT $LCD $CAM"         # всё вместе
  [S5]="$OPT $LCD $DIM"         # всё вместе, света в 10 раз меньше
)
for r in $RUNS; do
  $PY -m vit_optica.main $COMMON ${ARGS[$r]} ${EXTRA:-} --comment stand_$r${TAG:-} 2>&1 \
    | tee "$OUT/log_stand_$r${TAG:-}.txt"
done
