# Dashboard 生成说明
## 1. 需要修改的python包(以下包都需要通过 `pip install -e .`的方法安装)

### `SAEDashboard`
修改文件`/home/notebook/code/group/xuekaiwen/mask_diffusion/interpretable_models/remote/SAEDashboard/sae_dashboard/neuronpedia/neuronpedia_vector_runner.py`
具体变动内容均用注释表示

### `SAELens`

修改文件`/home/notebook/code/group/xuekaiwen/mask_diffusion/interpretable_models/remote/SAELens/sae_lens/training/activations_store.py`

### `transformer-lens`

之前修改过，沿用之前的修改，不变动

### 安装命令

- 安装SAEDashboard
```bash
cd remote/SAEDashboard
pip install -e .
```
- 安装sae-lens(在SAELens-5.11.0目录下安装)
```bash
cd remote/SAELens-5.11.0
pip install -e .
```
- 安装transformer-lens.(因sae-lens用 `pip install -e .` 安装时，会自动安装transformer-lens，而transformer-lens我们也有修改，所以请先安装sae_lens，再安装transformer-lens.)
```bash
cd remote/transformer-lens
pip install -e .
```

## 2. 运行代码生成特定层的特征向量

例如，生成 **第 0 层** 的特征：

```bash
cd /home/notebook/code/group/xuekaiwen/mask_diffusion/interpretable_models/remote/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils
export HF_ENDPOINT=https://hf-mirror.com
python generate-dashboards-as-vectors.py \
    --creator-name='AxBench Team' \
    --release-id=axbench \
    --release-title='AxBench Paper' \
    --url=https://github.com/stanfordnlp/axbench \
    --model-name="Llama-3-8B" \
    --model-dtype=bfloat16 \
    --neuronpedia-source-set-id=debug-20251013-vector-promptnums\
    --neuronpedia-source-set-description='Residual Stream - 16k' \
    --hook-point=mlp.hook_in \
    --source-dtype=bfloat16 \
    --layer-num=0 \
    --prompts-huggingface-dataset-path="wikimedia/wikipedia:20231101.en" \
    --activation-thresholds-json-file=activation-thresholds-new.json \
    --n-prompts-total=24576 \
    --n-tokens-in-prompt=64 \
    --n-prompts-per-batch=64 \
    --include-original-vectors-in-output

# 并行版本
mkdir -p logs out
source ~/.bashrc 2>/dev/null || true
conda activate dashboard

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_ENDPOINT=https://hf-mirror.com
LAYER_BASE=8   # 这次从第 8 层开始

for i in $(seq 0 1); do
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
    --prompts-huggingface-dataset-path="wikimedia/wikipedia:20231101.en" \
    --activation-thresholds-json-file=activation-thresholds-new.json \
    --n-prompts-total=24576 \
    --n-tokens-in-prompt=64 \
    --n-prompts-per-batch=64 \
    --include-original-vectors-in-output \
    --output-dir "out" \
    >"logs/layer-$LAYER_NUM.log" 2>&1 &
done

wait
echo "[INFO] Layers 8..15 finished."

```

参数说明：

* `--model-name`：模型名（与 `MODEL_MAP` 对应）
* `--layer-num`：要生成的层号
* `--hook-point`：Hook 位置，我们是`mlp.hook_in`
* `--n-prompts-total`：总提示数量，越大越准，目前24576就比较准了，如果资源充足，可以考虑更大
* `--n-prompts-per-batch`：每批提示数量
* `--prompts-huggingface-dataset-path`：所使用的数据集，用`wikimedia/wikipedia:20231101.en`会比较好，因为数据干净

备注：

只需要修改`layer-num`参数即可。

---

## 3. 转换 Neuronpedia JSON 文件格式

使用脚本 **`run_json_tran.sh`** 批量将 Neuronpedia 生成的特征 JSON 文件转换为当前所需格式。

记得修改`BASE_DIR`

```bash
bash run_json_tran.sh exports-processing-debug/Llama-3-8B/debug-20251011-vector-promptnums

```