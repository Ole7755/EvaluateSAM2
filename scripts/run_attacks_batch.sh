#!/usr/bin/env bash
# 批量运行 SAM2 对抗攻击实验。

set -euo pipefail

ATTACKS=(fgsm pgd bim cw)

SEQS=(
  bear
  bmx-bumps
  boat
  boxing-fisheye
  breakdance-flare
  bus
  car-turn
  cat-girl
  classic-car
  color-run
  crossing
  dance-jump
  dancing
  disc-jockey
  dog-agility
  dog-gooses
  dogs-scale
  drift-turn
  drone
  elephant
  flamingo
  hike
  hockey
  horsejump-low
  kid-football
  kite-walk
  koala
  lady-running
  lindy-hop
  longboard
  lucia
  mallard-fly
  mallard-water
  miami-surf
  motocross-bumps
  motorbike
  night-race
  paragliding
  planes-water
  rallye
  rhino
  rollerblade
  schoolgirls
  scooter-board
  scooter-gray
  sheep
  skate-park
  snowboard
  soccerball
  stroller
  stunt
  surf
  swing
  tennis
  tractor-sand
  train
  tuk-tuk
  upside-down
  varanus-cage
  walking
)

declare -A GT_LABELS=(
  [bear]=1
  [bmx-bumps]=1
  [boat]=1
  # 如需覆盖不同序列的首帧标签，请在此补充键值对。
)

run_attack() {
  local sequence=$1
  local attack=$2
  local frame_token=${3:-00000}
  local gt_label=${GT_LABELS[$sequence]:-1}

  local metrics_path="outputs/${sequence}/${attack}/${frame_token}_${attack}_metrics.json"
  if [[ -f "${metrics_path}" ]]; then
    echo "[SKIP] ${sequence} ${attack} 已存在 ${metrics_path}"
    return
  fi

  local extra=()
  case "${attack}" in
    pgd|bim)
      extra+=(--random-start)
      ;;
    cw)
      extra+=(--cw-lr 0.01 --cw-confidence 0.0 --cw-binary-steps 5)
      ;;
  esac

  python3 scripts/run_uap_attack.py \
    --sequence "${sequence}" \
    --frame-token "${frame_token}" \
    --gt-label "${gt_label}" \
    --obj-id 1 \
    --attack "${attack}" \
    --epsilon 0.03 \
    --step-size 0.01 \
    --steps 40 \
    --input-size 1024 \
    --mask-threshold 0.5 \
    --device cuda \
    "${extra[@]}"
}

for seq in "${SEQS[@]}"; do
  for atk in "${ATTACKS[@]}"; do
    echo "[INFO] 运行 ${seq} - ${atk}"
    run_attack "${seq}" "${atk}"
  done
done

echo "[INFO] 全部攻击任务完成。"
