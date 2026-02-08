import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from datasets import Dataset
# /home/user/ondemand/program/verl/data/train/enigama/train/binario/en/train.jsonl
BASE_DIR = "/home/user/ondemand/program/verl/data/train/enigama"
OUTPUT_TRAIN = "puzzle-train"
OUTPUT_TEST  = "puzzle-test"
OUTPUT_PATH = '/home/user/ondemand/program/verl/data/train'


def pack_ground_truth(obj):
    return json.dumps({"__wrapped__": True, "value": obj}, ensure_ascii=False)

def to_dict(x):
    try:
        return json.loads(x) if isinstance(x, str) else x
    except Exception as e:
        print(e)
        return x   # 如果不是合法 JSON，就原样返回

def load_split(split): 
    root = os.path.join(BASE_DIR, split)
    jsonl_files = glob(f"{root}/**/en/*.jsonl", recursive=True)

    rows = []
    idx = 0
    for file in tqdm(jsonl_files, desc=f"Processing {split}"):
        task_name = file.split("/")[-3]   # 目录名作为task_name

        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt = data["prompt"]
                answer = data["answer"]
                ability = data["ability"]
                meta = data["meta"]
                if type(meta) == dict:
                    meta = json.dumps(meta, ensure_ascii=False)

                rows.append({
                    "prompt": [
                        {"content": prompt, "role": "user"}
                    ],
                    "ability": task_name,
                    "data_source": f"PUZZLE",
                    "reward_model": {
                        "ground_truth": meta,
                        "style": "rule"
                    },
                    "extra_info": {
                        "index": idx,
                        "split": split,
                        "answer": pack_ground_truth(answer),
                        'task_name': task_name
                    }
                })
                idx += 1

    return rows


if __name__ == "__main__":
    train_df = load_split("train")
    test_df  = load_split("test")
    
    k = int(len(train_df)/1000)
    example = train_df[0]
    with open(os.path.join(OUTPUT_PATH, f"{OUTPUT_TRAIN}-{k}k.json"), "w") as f:
        json.dump(example, f, indent=2)
    dataset = Dataset.from_list(train_df)
    dataset.to_parquet(os.path.join(OUTPUT_PATH, f'{OUTPUT_TRAIN}-{k}k.parquet'))
    

    dataset = Dataset.from_list(test_df)
    k = len(test_df)
    example = dataset[0]
    with open(os.path.join(OUTPUT_PATH, f"{OUTPUT_TEST}-{k}.json"), "w") as f:
        json.dump(example, f, indent=2)
    dataset.to_parquet(os.path.join(OUTPUT_PATH, f'{OUTPUT_TEST}-{k}.parquet'))
    