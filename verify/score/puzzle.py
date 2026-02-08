import json
import os
import importlib
import pandas as pd
import concurrent.futures
import argparse

# Register all available task verifiers
registried_tasks = {}
def registried():
    try:
        folder_path = '/home/wxy320/ondemand/program/verify/score/puzzle_tasks'
        for task in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, task)):
                try:
                    module = importlib.import_module(f'score.puzzle_tasks.{task}.verifier')
                    registried_tasks[task] = module.verify
                except Exception:
                    continue
    except Exception:
        print('Cannot find verifiable_tasks directory')

def compute_score(solution_str, meta, task_name, answer) -> float:
    if len(registried_tasks) == 0: registried()
    if isinstance(meta, str):
        meta = json.loads(meta)
    try:
        verify_fn = registried_tasks[task_name]
        score = verify_fn(solution_str, answer, meta)
        # if score != 1: score=-1
        return score
    except Exception:
        return -1