#!/usr/bin/env bash
set -euo pipefail
ROOT="/opt/data/private/task/daily3/semanvar/HART"
cd "$ROOT"
EXP_NAME="03_collapse_s36_l5"
GPU_ID="4"
RUN_ROOT="/opt/data/private/task/daily3/semanvar/HART/output/geneval_hart_rewrite_7exp_20260510"
IMAGE_DIR="$RUN_ROOT/$EXP_NAME/images"
RESULT_DIR="$RUN_ROOT/$EXP_NAME/results"
mkdir -p "$RESULT_DIR" "$RUN_ROOT/score_logs"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
PYTHON="/root/miniconda3/envs/gen/bin/python"
CONFIG="/root/miniconda3/envs/gen/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
MODEL_PATH="/opt/data/private/task/checkpoint"
echo "[hart-score] exp=$EXP_NAME gpu=$GPU_ID image_dir=$IMAGE_DIR"
echo "[hart-score] start $(date)"
"$PYTHON" -u evaluation/gen_eval/evaluate_images.py \
  --imagedir "$IMAGE_DIR" \
  --outfile "$RESULT_DIR/results.jsonl" \
  --model-config "$CONFIG" \
  --model-path "$MODEL_PATH"
"$PYTHON" evaluation/gen_eval/summary_scores.py "$RESULT_DIR/results.jsonl" > "$RESULT_DIR/summary.txt"
cat "$RESULT_DIR/summary.txt"
echo "[hart-score] done $(date)"
