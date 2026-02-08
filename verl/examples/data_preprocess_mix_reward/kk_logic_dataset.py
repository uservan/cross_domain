""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import datasets
from datasets import concatenate_datasets

def make_prefix(dp, template_type):
    quiz = dp['quiz']
    if template_type == 'mix_reward':
        prefix = f"""The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\n{quiz}"""
    elif template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/train')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='K-and-K/knights-and-knaves')
    parser.add_argument('--train_size', type=int, default=5)
    parser.add_argument('--test_size', type=int, default=700)
    parser.add_argument('--template_type', type=str, default='mix_reward')
    
    args = parser.parse_args()
    
    data_source = 'kk_logic'
    TRAIN_SIZE = args.train_size * 1000
    TEST_SIZE = args.test_size
    
    train_dataset = datasets.load_dataset(args.data_path, 'train')
    train_dataset = concatenate_datasets([train_dataset[s] for s in train_dataset.keys()])
    train_dataset = train_dataset.select(range(TRAIN_SIZE))
    print(len(train_dataset))

    test_dataset = datasets.load_dataset(args.data_path, 'test')
    test_dataset = concatenate_datasets([test_dataset[s] for s in test_dataset.keys()])
    test_dataset = test_dataset.select(range(TEST_SIZE))
    print(len(test_dataset))
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "solution_text_format": example['solution_text_format'],
                "statements": example['statements']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, f'kk_Logic_{args.train_size}k.parquet'))
    example = train_dataset[0]
    with open(os.path.join(local_dir, f"kk-Logic-{args.train_size}k.json"), "w") as f:
        json.dump(example, f, indent=2)
    test_dataset.to_parquet(os.path.join(local_dir, f'kk_Logic_{args.test_size}.parquet'))
    example = test_dataset[0]
    with open(os.path.join(local_dir, f"kk-Logic-{args.test_size}.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)