import re
import os
import datasets
import random
from typing import Any, Dict, List, Optional

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import pickle
from datasets import Dataset


def get_multiple_choice_answers(data):
    answers = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    random.shuffle(answers)

    # Map options to letters
    options = ["A", "B", "C", "D"]
    options_to_answers = {
        letter: answer for letter, answer in zip(options, answers)
    }

    # Format the options into the string
    multiple_choice_string = ", ".join(
        f"{letter}) {options_to_answers[letter]}" for letter in options
    )

    # Save the letter corresponding to the correct answer
    correct_answer_letter = next(
        letter
        for letter, answer in options_to_answers.items()
        if answer == data["Correct Answer"]
    )

    return multiple_choice_string, correct_answer_letter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/test_data')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'Idavidrein/gpqa'

    train_dataset = datasets.load_dataset(data_source,'gpqa_diamond', split='train')
    print("\n\nOriginal dataset: ", "\n", train_dataset)
    
    instruction_following = 'Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.'

     # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("Question")
            multiple_choice_string, correct_answer_letter = get_multiple_choice_answers(example)
            data = {
                "data_source": 'gpqa-science',
                "prompt": [
                    {
                        "role": "user",
                        "content": instruction_following + "\n\n" + question + "\n" + multiple_choice_string,
                    }
                ],
                "ability": "science",
                "reward_model": {"style": "rule", "ground_truth": correct_answer_letter},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data
        return process_fn

    # 再做 map 结构转换
    train_dataset = train_dataset.map(function=make_map_fn("test"), with_indices=True,remove_columns=train_dataset.column_names)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, f"gpqa.parquet"))
    example = train_dataset[0]
    with open(os.path.join(local_save_dir, f"gpqa.json"), "w") as f:
        json.dump(example, f, indent=2)
        
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)

    

