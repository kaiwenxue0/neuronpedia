import json
import os
import argparse
from pathlib import Path

def quantile_bucket_name(i, total):
    if i == 0:
        return "Top 1%"
    elif i == total - 1:
        return "Bottom 1%"
    else:
        return f"Subsample interval {i}"

def convert_feature_to_json(feature, num_buckets=10):
    activations = feature.get("activations", [])
    feature_index = feature.get("feature_index", -1)

    # 为每个 activation 计算 max 激活值
    scored_acts = []
    for act in activations:
        values = act.get("values", [])
        max_val = max(values) if values else float('-inf')
        scored_acts.append((max_val, act))

    # 按 max_val 降序排序
    scored_acts.sort(reverse=True, key=lambda x: x[0])
    total = len(scored_acts)
    base_size = total // num_buckets
    leftover = total - base_size * (num_buckets - 1)  # 剩余的样本数

    examples_quantiles = []

    for i in range(num_buckets):
        if i == 5:
            # 第5组装余数
            start = i * base_size
            end = start + leftover
        elif i < 5:
            start = i * base_size
            end = (i + 1) * base_size
        else:
            # i > 5 的组要右移（因为第5组占了多个）
            start = (i - 1) * base_size + leftover
            end = start + base_size

        bucket = scored_acts[start:end]
        examples = []

        for _, act in bucket:
            values = act.get("values", [])
            tokens = act.get("tokens", [])
            tokens, values = clean_tokens_and_acts(tokens, values)
            if values:
                token_ind = int(max(range(len(values)), key=lambda i: values[i]))
            else:
                token_ind = -1

            examples.append({
                "tokens_acts_list": values,
                "train_token_ind": token_ind,
                "is_repeated_datapoint": False,
                "tokens": tokens,
                "ha_haiku35_resampled": None
            })

        if examples:
            examples_quantiles.append({
                "quantile_name": quantile_bucket_name(i, num_buckets),
                "examples": examples
            })


    # 汇总 min/max
    all_values = [v for _, act in scored_acts for v in act.get("values", [])]
    act_min = min(all_values) if all_values else 0
    act_max = max(all_values) if all_values else 0

    return {
        "index": int(feature_index),
        "examples_quantiles": examples_quantiles,
        "top_logits": feature.get("pos_str", [])[:5],
        "bottom_logits": feature.get("neg_str", [])[:5],
        "act_min": act_min,
        "act_max": act_max
    }

def clean_tokens_and_acts(tokens, values):
    """
    将 tokens 中的 '\n' 替换为空字符串；
    如果替换后整个 token 为空，则剔除该 token，
    并同步剔除对应位置的 activation。
    """
    new_tokens, new_values = [], []
    L = min(len(tokens), len(values))  # 防止长度不一致
    for i in range(L):
        t, v = tokens[i], values[i]
        if isinstance(t, str):
            cleaned = t.replace("\n", "")  # 只去掉换行符
            if cleaned != "":  # 如果替换后不为空，则保留
                new_tokens.append(cleaned)
                new_values.append(v)
        else:
            new_tokens.append(t)
            new_values.append(v)
    return new_tokens, new_values



def main():
    """Main entry: Read features from JSON file and batch convert to new format"""

    # ====== Command-line arguments ======
    parser = argparse.ArgumentParser(description="Convert feature JSON files to a specific format.")
    parser.add_argument("--input_file", type=str, default="Llama-3-8B_blocks.0.mlp.hook_in/batch-0.json", help="Path to the input JSON file containing a 'features' list.")
    parser.add_argument("--output_dir", type=str, default="Llama-3-8B_blocks.0.converted", help="Directory to save converted JSON files (default: llama3_converted_features).")
    parser.add_argument("--num_buckets", type=int, default=10, help="Number of buckets for conversion (default: 10, meaning 1 top + 8 subsample + 1 bottom).")

    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    # ====== Load input data ======
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ====== Create output directory ======
    os.makedirs(output_dir, exist_ok=True)

    # ====== Process each feature ======
    processed_count = 0
    layer_num = data.get("layer", -1)
    for feature in data.get("features", []):
        feature_index = feature.get("feature_index")
        if feature_index is None:
            continue  # Skip if no feature index

        # Convert the feature
        converted_json = convert_feature_to_json(feature, num_buckets=args.num_buckets)

        # Save to file
        combined_index = layer_num * 1_000_000 + feature_index
        output_json_name = f"{combined_index}"
        output_path = os.path.join(output_dir, f"{combined_index}.json")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(converted_json, f_out, indent=2, ensure_ascii=False)

        processed_count += 1

    print(f"Conversion completed. {processed_count} files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
