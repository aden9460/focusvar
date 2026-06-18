#!/usr/bin/env bash
set -euo pipefail
ROOT="/opt/data/private/task/daily3/semanvar/HART"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PATH="/root/miniconda3/envs/hart121/bin:$PATH"
export CUDA_HOME="/root/miniconda3/envs/hart121"
export LD_LIBRARY_PATH="/root/miniconda3/envs/hart121/lib:/root/miniconda3/envs/hart121/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/root/miniconda3/envs/hart121/bin/python"
MODEL_PATH="/opt/data/private/task/hart_pretrained/hart-0.7b-1024px/llm"
TEXT_MODEL_PATH="/opt/data/private/task/hart_pretrained/Qwen2-VL-1.5B-Instruct"
METADATA_FILE="$ROOT/evaluation/gen_eval/prompts/evaluation_metadata.jsonl"
PROMPT_REWRITE_CACHE="/opt/data/private/task/daily3/semanvar/Infinity/evaluation/gen_eval/prompt_rewrite_cache.json"
RUN_ROOT="/opt/data/private/task/daily3/semanvar/HART/output/geneval_hart_rewrite_7exp_20260510"
EXP_NAME="03_collapse_s36_l5"
GPU_ID="4"
OUTDIR="$RUN_ROOT/$EXP_NAME/images"
LOGDIR="$RUN_ROOT/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[hart-exp] exp=$EXP_NAME gpu=$GPU_ID out=$OUTDIR"
echo "[hart-exp] start $(date)"
"$PYTHON" -u evaluation/gen_eval/infer4eval.py \
  --model_path "$MODEL_PATH" \
  --text_model_path "$TEXT_MODEL_PATH" \
  --seed 1 \
  --use_ema 1 \
  --cfg 4.5 \
  --more_smooth True \
  --n_samples 4 \
  --metadata_file "$METADATA_FILE" \
  --rewrite_prompt 1 \
  --load_rewrite_prompt_cache 1 \
  --prompt_rewrite_cache_file "$PROMPT_REWRITE_CACHE" \
  --outdir "$OUTDIR" \
  --enable_fastvar_compute_merge 0 --enable_spacevar_compute_merge 0 --enable_layerwise_cond_only_collapse 1 --cond_only_start_scale 36 --cond_only_start_layer 5
echo "[hart-exp] done $(date)"
