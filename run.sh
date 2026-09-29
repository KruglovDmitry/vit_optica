#!/usr/bin/env bash
# =====================================================================
#  Весь план экспериментов ViT одной командой (последовательно, как в optic_train).
#
#    bash vit_optica/run.sh probe       # 0. замер скорости и прогноз времени для фазы scratch
#    bash vit_optica/run.sh scratch     # 1. обучение С НУЛЯ: цифра (несколько сидов) и оптика
#    bash vit_optica/run.sh cross       # 1б. перекрёстная оценка весов, обученных на оптике
#    bash vit_optica/run.sh             # все фазы, кроме probe
#    bash vit_optica/run.sh digital     # цифровое обучение со случайным порядком патчей
#    bash vit_optica/run.sh infer       # 2. инференс с оптикой на цифровых весах
#    bash vit_optica/run.sh order       # 3. порядок токенов: raster против random
#    bash vit_optica/run.sh layers      # 4. число оптических блоков (как таблица CIFAR-10)
#    bash vit_optica/run.sh noise       # 5. шум σ на точном умножении (проверка κ)
#    bash vit_optica/run.sh finetune    # 6. дообучение с оптикой
#    DRY=1 bash vit_optica/run.sh       # только напечатать команды
#
#  Рабочая папка — родитель vit_optica/ (рядом лежит Optical_matrix_multiplication/).
# =====================================================================
set -u
GPUS=${GPUS:-"4 5 6 7"}
read -r -a GPU_ARR <<< "$GPUS"; NGPU=${#GPU_ARR[@]}
CVD=$(IFS=,; echo "${GPU_ARR[*]}"); SIM_DEVICES=$(seq -s, 0 $((NGPU - 1)))

PY=${PY:-python}
DATASET=${DATASET:-imagenette}    # imagenette | imagewoof | tinyimagenet | cifar10 ...
DATA=${DATA:-./data}
OUT=${OUT:-./runs_vit}
SEED=${SEED:-1337}
EPOCHS=${EPOCHS:-300}          # обучение с нуля (цифра и оптика — одинаково)
FT_EPOCHS=${FT_EPOCHS:-10}     # дообучение с оптикой
BATCH=${BATCH:-128}
EVAL_BATCH=${EVAL_BATCH:-128}  # одинаковый во всех запусках
LENSES=${LENSES:-"16384 32768"}
# обучение с нуля: одинаковый рецепт для цифры и оптики
SEEDS_DIGITAL=${SEEDS_DIGITAL-"1337 1 2"}  # пустая строка = без цифры      # разброс цифрового эталона
FORMULAS=${FORMULAS-"shift split"}          # пустая строка = без оптики
SCRATCH_WHERE=${SCRATCH_WHERE:-"qk+av ff+proj all"}
SCRATCH_LENSES=${SCRATCH_LENSES:-"16384"}
PROBE_ITERS=${PROBE_ITERS:-20}
SIM_PARALLEL=${SIM_PARALLEL:-1}
DRY=${DRY:-0}
EXTRA=${EXTRA:-}               # любые доп. аргументы main.py для всех запусков
# модель: ViT-Tiny по ширине, MLP×2, чтобы все матрицы (макс. 384) помещались в поле 512
MODEL="--h_dim 192 --depth 12 --heads 6 --mlp_ratio 2 --aperture 512 --distance 0.15"

SP=""
if [[ "$SIM_PARALLEL" == "1" && "$NGPU" -ge 2 ]]; then
  SP="--sim_parallel 1 --sim_devices $SIM_DEVICES --check_parallel 1"
fi
mkdir -p "$OUT"

launch() {   # launch <comment> <args...>
  local c="$1"; shift
  local cmd="CUDA_VISIBLE_DEVICES=$CVD $PY -m vit_optica.main --dataset $DATASET \
    --data_dir $DATA --out_dir $OUT --seed ${OVERRIDE_SEED:-$SEED} $MODEL \
    --batch_size $BATCH --eval_batch_size $EVAL_BATCH $EXTRA $* --comment $c"
  if [[ "$DRY" == "1" ]]; then echo "$cmd"; else
    echo ">>> [$(date +%H:%M:%S)] $c"; eval "$cmd 2>&1 | tee $OUT/log_${c}.txt"; fi
}
# цифровые веса: raster — первый сид фазы scratch; random — фаза digital
ck() { if [[ "$1" == raster ]]; then echo "$OUT/ckpt_digital_scr_digital_s${SEED}.pt";
       else echo "$OUT/ckpt_digital_digi_$1.pt"; fi; }
need() { [[ -f "$1" || "$DRY" == "1" ]] || { echo "!! нет $1 — сначала фаза digital"; return 1; }; }

RECIPE="--epochs $EPOCHS --warmup_epochs 10 --lr 1e-3"

phase_probe() {   # сколько займёт каждый запуск фазы scratch
  launch "probe_digital" --mode digital $RECIPE --probe_iters $PROBE_ITERS --download 1
  for F in $FORMULAS; do for W in $SCRATCH_WHERE; do for L in $SCRATCH_LENSES; do
    launch "probe_${F}_${W//+/_}_L${L}" --mode $F --optic_where $W --lens_size $L \
      $RECIPE --probe_iters $PROBE_ITERS $SP
  done; done; done
}

phase_scratch() {
  for S in $SEEDS_DIGITAL; do
    OVERRIDE_SEED=$S launch "scr_digital_s${S}" --mode digital $RECIPE --download 1
  done
  for F in $FORMULAS; do for W in $SCRATCH_WHERE; do for L in $SCRATCH_LENSES; do
    launch "scr_${F}_${W//+/_}_L${L}" --mode $F --optic_where $W --lens_size $L $RECIPE $SP
  done; done; done
}

phase_cross() {   # веса, обученные на оптике: в цифре, с другой апертурой; цифровые веса — с той же оптикой
  local dig="$(ck raster)"
  for F in $FORMULAS; do for W in $SCRATCH_WHERE; do for L in $SCRATCH_LENSES; do
    local tag="${F}_${W//+/_}_L${L}" c="$OUT/ckpt_${F}_scr_${F}_${W//+/_}_L${L}.pt"
    need "$c" || continue
    launch "cross_${tag}_as_digital" --mode digital --eval_only 1 --load_ckpt "$c"
    for L2 in $LENSES; do
      [[ "$L2" == "$L" ]] && continue
      launch "cross_${tag}_at_L${L2}" --mode $F --optic_where $W --lens_size $L2 \
        --eval_only 1 --load_ckpt "$c" $SP
    done
    need "$dig" && launch "cross_digital_with_${tag}" --mode $F --optic_where $W \
      --lens_size $L --eval_only 1 --load_ckpt "$dig" $SP
  done; done; done
}

phase_digital() {
  # raster-веса дают фаза scratch (сид $SEED); здесь — только случайный порядок патчей
  launch "digi_random" --mode digital --token_order random $RECIPE --download 1
}

phase_infer() {   # формула × место оптики × апертура, веса цифровые
  need "$(ck raster)" || return
  for F in shift split; do
    for W in ff proj ff+proj qk av qk+av all; do
      for L in $LENSES; do
        launch "inf_${F}_${W//+/_}_L${L}" --mode $F --optic_where $W --lens_size $L \
          --eval_only 1 --load_ckpt "$(ck raster)" $SP
      done
    done
  done
  # нормировка split по всему батчу (как в старых запусках) — проверка утечки между примерами
  launch "inf_split_all_L16384_global" --mode split --optic_where all --lens_size 16384 \
    --split_norm global --eval_only 1 --load_ckpt "$(ck raster)" $SP
}

phase_order() {   # важна ли структура смешивания соседних позиций
  need "$(ck random)" || return
  for F in shift split; do
    for W in qk+av all; do
      launch "ord_random_${F}_${W//+/_}" --mode $F --optic_where $W --lens_size 16384 \
        --eval_only 1 --load_ckpt "$(ck random)" $SP
    done
  done
}

phase_layers() {  # аналог таблицы CIFAR-10: 1/3/6/12 блоков с оптикой (оптика во внимании)
  need "$(ck raster)" || return
  for N in 1 3 6 12; do
    launch "layers_split_attn_N${N}" --mode split --optic_where qk+av --optic_layers $N \
      --lens_size 16384 --eval_only 1 --load_ckpt "$(ck raster)" $SP
  done
}

phase_noise() {   # точное умножение с шумом σ: ошибка/σ ≈ κ, сравнение shift и split
  need "$(ck raster)" || return
  for F in shift split; do
    for S in 0.001 0.01 0.1; do
      launch "noise_${F}_ff_proj_s${S}" --mode $F --optic_where ff+proj --stub_sim \
        --noise_sigma $S --eval_only 1 --load_ckpt "$(ck raster)"
    done
  done
}

phase_finetune() {
  need "$(ck raster)" || return
  for F in shift split; do
    for W in ff+proj all; do
      for L in $LENSES; do
        launch "ft_${F}_${W//+/_}_L${L}" --mode $F --optic_where $W --lens_size $L \
          --load_ckpt "$(ck raster)" --epochs $FT_EPOCHS --warmup_epochs 1 \
          --lr 1e-4 --min_lr 1e-6 --eval_every 1 $SP
      done
    done
  done
}

case "${1:-all}" in
  probe) phase_probe ;; scratch) phase_scratch ;; cross) phase_cross ;;
  digital) phase_digital ;; infer) phase_infer ;; order) phase_order ;;
  layers) phase_layers ;; noise) phase_noise ;; finetune) phase_finetune ;;
  all) phase_scratch; phase_cross; phase_digital; phase_infer; phase_order; phase_layers;
       phase_noise; phase_finetune ;;
  *) echo "usage: bash run.sh [all|probe|scratch|cross|digital|infer|order|layers|noise|finetune]"; exit 1 ;;
esac
echo "ГОТОВО. Сводка: $PY -m vit_optica.collect_results $OUT $OUT/summary.csv"
