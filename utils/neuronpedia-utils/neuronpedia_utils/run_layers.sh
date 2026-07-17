#!/usr/bin/env bash
set -euo pipefail


# ===== 参数与默认值 =====
# 优先级：位置参数 $1 > 环境变量 LAYER_BASE > 默认 8
LAYER_BASE="${1:-${LAYER_BASE:-8}}"
# 并行层数（默认 8），也可用 $2 或环境变量 LAYER_COUNT 覆盖
LAYER_COUNT="${2:-${LAYER_COUNT:-8}}"

# 校验为非负整数
[[ "$LAYER_BASE" =~ ^[0-9]+$ ]] || { echo "LAYER_BASE 必须是非负整数，当前：$LAYER_BASE"; exit 1; }
[[ "$LAYER_COUNT" =~ ^[0-9]+$ ]] || { echo "LAYER_COUNT 必须是非负整数，当前：$LAYER_COUNT"; exit 1; }
if (( LAYER_COUNT == 0 )); then echo "LAYER_COUNT 不能为 0"; exit 1; fi

echo "[INFO] LAYER_BASE=$LAYER_BASE, LAYER_COUNT=$LAYER_COUNT (将运行层 $LAYER_BASE..$((LAYER_BASE+LAYER_COUNT-1)))"

    
mkdir -p logs out


export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4


for i in $(seq 0 $((LAYER_COUNT-1))); do
  LAYER_NUM=$((LAYER_BASE + i))
  echo "[INFO] Launching layer $LAYER_NUM on GPU $i ..."
  CUDA_VISIBLE_DEVICES="$i" \
  python generate-dashboards-as-vectors.py \
    --creator-name="AxBench Team" \
    --release-id=axbench \
    --release-title="AxBench Paper" \
    --url=https://github.com/stanfordnlp/axbench \
    --model-name="Llama-3-8B" \
    --model-dtype=bfloat16 \
    --neuronpedia-source-set-id=debug-20251013-vector-promptnums \
    --neuronpedia-source-set-description="Residual Stream - 16k" \
    --hook-point=mlp.hook_in \
    --source-dtype=bfloat16 \
    --layer-num="$LAYER_NUM" \
    --prompts-huggingface-dataset-path="/home/notebook/code/group/xuekaiwen/mask_diffusion/interpretable_models/data/wikimedia/wikipedia:20231101.en" \
    --activation-thresholds-json-file=activation-thresholds-new.json \
    --n-prompts-total=24576 \
    --n-tokens-in-prompt=64 \
    --n-prompts-per-batch=64 \
    --include-original-vectors-in-output \
    --output-dir "out/layer-$LAYER_NUM" \
    >"logs/layer-$LAYER_NUM.log" 2>&1 &
done

wait
echo "[INFO] Layers $LAYER_BASE..$((LAYER_BASE+LAYER_COUNT-1)) finished."
