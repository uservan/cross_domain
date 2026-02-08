# stage1_generate_vllm.py
import os
import json
import argparse
from typing import List, Dict, Any

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from score import default_compute_score

project_root = "/home/wxy320/ondemand/program/verify"


def get_category_data(category: str, max_samples: int = -1) -> List[Dict[str, Any]]:
    if category == "skywork_math":  # math
        path = os.path.join(project_root, "test_data/MATH-500.parquet")
    elif category == "PUZZLE":
        path = os.path.join(project_root, "test_data/puzzle-test-395.parquet")
    elif category == "science":
        path = os.path.join(project_root, "test_data/gpqa.parquet")
    elif category == "kk":
        path = os.path.join(project_root, "test_data/kk_Logic_700.parquet")
    else:
        raise ValueError(f"Unknown category: {category}")

    df = pd.read_parquet(path)
    results = df.to_dict(orient="records")

    if max_samples > 0:
        results = results[:max_samples]
    return results


def count_existing_records(save_path: str) -> int:
    if not os.path.exists(save_path):
        return 0
    n = 0
    with open(save_path, "r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/scratch/pioneer/jobs/wxy320/verl/model/MIX-REWARD-GRPO-SMALL-Normal-merge/Qwen3-4B-Base-science-puzzle-math",
        help="HF repo id or local path for vLLM",
    )
    parser.add_argument("--dataset", default="science", choices=["skywork_math", "PUZZLE", "science", "kk"])
    parser.add_argument("--max_samples", type=int, default=3)

    parser.add_argument("--max_tokens", type=int, default=1024 * 16)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)

    # ✅ NEW: number of generations per problem
    parser.add_argument("--n", type=int, default=5, help="number of generations per problem")

    parser.add_argument("--save_path", type=str, default="/scratch/pioneer/jobs/wxy320/verl/base")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(
        args.save_path,
        f"{os.path.basename(args.model)}_{args.dataset}_n{args.n}.jsonl",
    )

    # ========================
    # Resume logic (correct for n>1)
    # ========================
    existing_records = count_existing_records(save_path)
    finished_samples = existing_records // args.n
    leftover = existing_records % args.n

    if existing_records > 0:
        print(f"🧭 Found {existing_records} existing records in {save_path}")
        print(f"✅ Interpreted as {finished_samples} fully finished samples (n={args.n}).")
        if leftover != 0:
            print(
                f"⚠️ Warning: {leftover} leftover records (incomplete last sample). "
                f"Will resume from sample {finished_samples} and regenerate that sample fully."
            )

    # ========================
    # Load data
    # ========================
    rows = get_category_data(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(rows)} samples from {args.dataset}")

    if finished_samples >= len(rows):
        print("🎉 All samples already processed.")
        return

    # ========================
    # Tokenizer for chat template
    # ========================
    # NOTE: use the base tokenizer for consistent prompting
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    # If leftover != 0, we "rewind" one sample to avoid misalignment.
    # E.g., if last sample had only 2/5 records written, we regenerate all 5 for that sample.
    start_sample = finished_samples
    if leftover != 0 and start_sample > 0:
        # We want to regenerate the incomplete sample again.
        # But since we already counted it as unfinished (finished_samples excludes it),
        # start_sample is already correct; we just proceed from start_sample.
        pass

    print(f"Will generate from sample index {start_sample}/{len(rows)} (resume-safe)")

    # ========================
    # Build prompts
    # ========================
    prompts: List[str] = []
    for ex in rows[start_sample:]:
        # parquet uses: ex["prompt"] is usually a list of messages like [{"role":"user","content":...}, ...]
        # your data: ex["prompt"][0]["content"]
        user_content = ex["prompt"][0]["content"] # if isinstance(ex.get("prompt"), list) else ex["prompt"]

        messages = [{"role": "user", "content": user_content}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True,  # enable if you want think-mode
        )
        prompts.append(text)

    # ========================
    # Load vLLM
    # ========================
    print(f"🚀 Loading {args.model} with vLLM ...")
    llm = LLM(model=args.model, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,  # ✅ number of completions per prompt
    )

    outputs = llm.generate(prompts, sampling_params)

    # ========================
    # Write results (jsonl)
    # Each completion is one line.
    # ========================
    acc = 0
    total = 0

    with open(save_path, "a") as fout:
        for i, (ex, output) in enumerate(zip(rows[start_sample:], outputs)):
            user_content = ex["prompt"][0]["content"] # if isinstance(ex.get("prompt"), list) else ex["prompt"]
            gt = ex["reward_model"]["ground_truth"]

            for j, out in enumerate(output.outputs):
                gen_text = out.text

                score = default_compute_score(
                    args.dataset,
                    gen_text,
                    gt,
                    extra_info=ex.get("extra_info"),
                )

                correct = bool(score["acc"])   # ✅ 关键修复

                acc += int(correct)
                total += 1

                fout.write(json.dumps({
                    "len": int(len(gen_text)),
                    "correct": correct,
                    "problem": str(user_content),
                    "input_text": str(prompts[i]),
                    "generated_text": str(gen_text),
                    "answer": gt.tolist() if hasattr(gt, "tolist") else gt
                }, ensure_ascii=False) + "\n")

            fout.flush()

    print(f"✅ Done! Acc={acc/total:.4f} ({acc}/{total})")
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()