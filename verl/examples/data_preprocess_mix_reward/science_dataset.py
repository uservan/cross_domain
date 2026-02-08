# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import random
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution_str = solution_str.strip()

    # 1️⃣ 处理 \text{...} 格式
    match = re.search(r"\\text\{([^}]*)\}", solution_str)
    if match:
        content = match.group(1).strip()
        # # 如果包含多个答案（含 , 或空格分隔），则当做无效
        # if "," in content or " " in content:
        #     return None
        return content.upper()  # 保证输出统一为大写字母
    else:
        return solution_str.upper()

def has_single_answer(answer_raw):
    """
    返回 True 表示答案合法：
      ✅ 仅当提取出的答案是一个单独的字母（A-Z / a-z）
      ❌ 否则过滤掉
    """
    solution = extract_solution(answer_raw)

    # 去除空格、换行等
    solution = solution.strip()

    # 必须是长度为 1 的字符，并且是字母
    return len(solution) == 1 and solution.isalpha()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="data/train", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument('--k', default=5, type=int)

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "nvidia/OpenScienceReasoning-2"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    
    instruction_following = 'Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("input")
            answer_raw = example.pop("expected_answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": 'nvidia-science',
                "prompt": [
                    {
                        "role": "user",
                        "content":  instruction_following + "\n\n" + question.split("Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}.")[-1].strip(),
                    }
                ],
                "ability": "science",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data
        return process_fn

    # 先随机抽 8k（或 args.k * 2000 之类以防过滤掉太多）
    raw_indices = random.sample(range(len(train_dataset)), k=args.k * 1200)
    subset = train_dataset.select(raw_indices)

    # 先做过滤，只保留“唯一答案样本”
    subset = subset.filter(lambda x: has_single_answer(x["expected_answer"]))

    # 如果过滤后超出 5k → 再裁切前 5000 个
    subset = subset.select(range(args.k * 1000))

    # 再做 map 结构转换
    train_dataset = subset.map(function=make_map_fn("train"), with_indices=True,remove_columns=subset.column_names)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, f"nvidia-Science-{args.k}k.parquet"))
    example = train_dataset[0]
    with open(os.path.join(local_save_dir, f"nvidia-Science-{args.k}k.json"), "w") as f:
        json.dump(example, f, indent=2)
        
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
