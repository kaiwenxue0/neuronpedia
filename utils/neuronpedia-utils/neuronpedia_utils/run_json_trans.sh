#!/bin/bash
set -e

# 固定根目录
BASE_DIR="/mnt/workspace/xuekaiwen/mask_diffusion/interpretable_models"

# 检查是否传入子目录路径参数
if [ -z "$1" ]; then
    echo "Usage: bash run_convert.sh <subdir>"
    echo "Example: bash run_convert.sh exports-processing-debug/Llama-3-8B/debug-20251011-vector-promptnums"
    exit 1
fi

# 拼接完整路径
ROOT_DIR="$BASE_DIR/$1"
# 输入目录和输出目录
INPUT_DIR="$ROOT_DIR/Llama-3-8B_blocks.0.mlp.hook_in"
OUTPUT_DIR="$ROOT_DIR/Llama-3-8B_blocks.0.converted"

# 遍历输入目录中所有 batch*.json 文件
for file in "$INPUT_DIR"/batch*.json; do
    echo "Processing $file ..."
    python json_neuronpedia_to_current.py \
        --input_file "$file" \
        --output_dir "$OUTPUT_DIR"
done

echo "All batch files processed."
