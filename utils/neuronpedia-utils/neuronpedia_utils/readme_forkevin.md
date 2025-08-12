# Dashboard 使用说明

## 1. 修改层映射文件

在 `layer_dir_map.py` 中，确保 **transcoder 每一层的映射关系** 正确无误。

---

## 2. 修改 `neuronpedia_vector_runner.py`

编辑文件：

```
./anaconda3/envs/dashboard/lib/python3.10/site-packages/sae_dashboard/neuronpedia/neuronpedia_vector_runner.py
```

在文件中添加以下代码：

```python
MODEL_MAP = {
    "Llama-3-8B": "/mnt/xuekaiwen/models/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
}
self.model_id_path = MODEL_MAP[self.model_id]
```

**目的**：

* 运行代码时，只需在参数中传入 `"Llama-3-8B"`
* 调用 `HookedTransformer` 时，会自动映射为本地模型目录路径

> 提示：与 **Circuit-tracer** 一样，记得修改 `HookedTransformer` 代码，以支持该路径映射机制。

---

## 3. 修改 `generate-dashboards-as-vectors.py`

将文件中的：

```python
ROOT_DIR = Path("/mnt/xuekaiwen")
```

替换为你本地的根目录路径。

---

## 4. 运行代码生成特定层的特征向量

例如，生成 **第 20 层** 的特征：

```bash
python generate-dashboards-as-vectors.py \
    --creator-name='AxBench Team' \
    --release-id=axbench \
    --release-title='AxBench Paper' \
    --url=https://github.com/stanfordnlp/axbench \
    --model-name="Llama-3-8B" \
    --model-dtype=bfloat16 \
    --neuronpedia-source-set-id=axbench-reft-r1-res-16k \
    --neuronpedia-source-set-description='Residual Stream - 16k' \
    --hook-point=hook_mlp_out \
    --source-dtype=bfloat16 \
    --layer-num=20 \
    --prompts-huggingface-dataset-path=monology/pile-uncopyrighted \
    --n-prompts-total=8192 \
    --n-tokens-in-prompt=128 \
    --n-prompts-per-batch=128 \
    --include-original-vectors-in-output
```

参数说明：

* `--model-name`：模型名（与 `MODEL_MAP` 对应）
* `--layer-num`：要生成的层号
* `--hook-point`：Hook 位置
* `--n-prompts-total`：总提示数量
* `--n-prompts-per-batch`：每批提示数量

---

## 5. 转换 Neuronpedia JSON 文件格式

使用脚本 **`json_neuronpedia_to_current.py`** 将 Neuronpedia 生成的特征 JSON 文件转换为当前所需格式。

```bash
python json_neuronpedia_to_current.py \
    --input-file batch-0.json \
    --output-dir converted_feature
```

* `--input-file`：Neuronpedia 输出的原始 JSON 文件（例如 `batch-0.json`）
* `--output-dir`：转换后 JSON 文件的保存目录

---

## 6. 上传 Feature JSON 文件至 GitHub

仓库地址：
[https://github.com/lt-0123/dashbroad](https://github.com/lt-0123/dashbroad)

**建议的文件结构：**

```
./features/[Your model name]/[feature_id].json
```

**示例：**

```
./features/llama-3-8B/12000019.json
```


