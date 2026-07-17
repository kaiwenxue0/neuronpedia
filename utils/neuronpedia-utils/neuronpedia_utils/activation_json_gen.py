import os
import json

def json_files_to_dict(folder_path, output_file="activation-thresholds-new.json"):
    # 获取文件夹中的所有 .json 文件（不包含子文件夹）
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".json") and os.path.isfile(os.path.join(folder_path, f))]
    
    # 去掉后缀
    file_names = [os.path.splitext(f)[0] for f in file_list]
    
    # 按字符串排序
    file_names.sort()
    
    # 生成字典：文件名 -> 0
    file_dict = {name: 0 for name in file_names}
    
    # 保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(file_dict, f, ensure_ascii=False, indent=2)

    print(f"已保存到 {output_file}")

# 使用方法：替换为你的文件夹路径
path = "/mnt/workspace/xuekaiwen/mask_diffusion/interpretable_models/exports-processing/Llama-3-8B/axbench-reft-r1-res-16k/Llama-3-8B_blocks.0.converted"
json_files_to_dict(path)
