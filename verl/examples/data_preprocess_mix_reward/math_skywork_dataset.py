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

import re
import os
import datasets
import random

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import pickle
from datasets import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/train2')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--k', default=5, type=int)

    args = parser.parse_args()

    data_source = 'Skywork/Skywork-OR1-RL-Data'

    dataset = datasets.load_dataset(data_source)
    print("\n\nOriginal dataset: ", "\n", dataset)

    def process_ground_truth(item):
        if "reward_model" in item and "ground_truth" in item["reward_model"]:
            try:
                item["reward_model"]["ground_truth"] = json.loads(item["reward_model"]["ground_truth"])
            except:
                pass
        return item
    
    dataset= dataset.map(process_ground_truth)
    
    def filter_fn(example):  
        if 'extra_info' not in example or 'model_difficulty' not in example['extra_info']:
            return False 
        if example['ability'] == 'math':
            difficulty = example['extra_info']['model_difficulty'].get('DeepSeek-R1-Distill-Qwen-7B') # DeepSeek-R1-Distill-Qwen-7B
        if example['ability'] == 'code':
            difficulty = example['extra_info']['model_difficulty'].get('DeepSeek-R1-Distill-Qwen-7B')
            # return True
        if difficulty is None:
            return False
        if difficulty < 1 or difficulty > 15:
            return False
        return True
    dataset = dataset.filter(filter_fn)
    print("\n\nFiltered dataset: ", "\n", dataset)

    data_list = []
    for key in dataset:
        data_list.extend([item for item in dataset[key]])

    math_data_list = random.sample([item for item in data_list if item['ability'] == 'math'], k=args.k*1000)
    code_data_list = random.sample([item for item in data_list if item['ability'] == 'code'], k=args.k*1000+100)

    for i in range(len(math_data_list)):
        data_source = math_data_list[i]['data_source']
        math_data_list[i]['extra_info']['data_source'] = data_source
        math_data_list[i]['data_source'] = 'skywork_math'

    # save_p = """"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. The final answer must appear at the end, wrapped in <answer>```python\n# YOUR CODE HERE\n```</answer>."""
    # replace_p = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."""
    for i in range(len(code_data_list)):
        data_source = code_data_list[i]['data_source']
        code_data_list[i]['extra_info']['data_source'] = data_source
        # code_data_list[i]['prompt'][0]['content'] = code_data_list[i]['prompt'][0]['content'].replace(replace_p, save_p)
        code_data_list[i]['data_source'] = 'skywork_code'
        new_ground_truth = {}
        item = code_data_list[i]['reward_model']['ground_truth']
        for key in item:
            if item[key] is not None:
                new_ground_truth[key] = item[key]
        code_data_list[i]['reward_model']['ground_truth'] = new_ground_truth

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    math_dataset = Dataset.from_list(math_data_list)
    math_dataset.to_parquet(os.path.join(local_dir, f"skywork-Math-{args.k}k.parquet"))
    example = math_dataset[0]
    with open(os.path.join(local_dir, f"skywork-Math-{args.k}k.json"), "w") as f:
        json.dump(example, f, indent=2)

    code_data_test_list = code_data_list[-100:]
    code_data_list = code_data_list[:-100]
    code_dataset = Dataset.from_list(code_data_list)
    code_dataset.to_parquet(os.path.join(local_dir, f"skywork-Code-{args.k}k.parquet"))
    example = code_dataset[0]
    with open(os.path.join(local_dir, f"skywork-Code-{args.k}k.json"), "w") as f:
        json.dump(example, f, indent=2)
    code_dataset = Dataset.from_list(code_data_test_list)
    code_dataset.to_parquet(os.path.join(local_dir, f"skywork-Code-100.parquet"))
    example = code_dataset[0]
    with open(os.path.join(local_dir, f"skywork-Code-100.json"), "w") as f:
        json.dump(example, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    