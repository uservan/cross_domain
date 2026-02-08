# stage1_generate_vllm.py
import os, json, argparse
from tqdm import tqdm
from typing import Dict, Any, List
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd

project_root = '/home/user/ondemand/program/verl'

def get_category_data(category: str, split: str = "test", max_samples: int = -1):
    results = []
    if category == "HuggingFaceH4/MATH-500":
        path = os.path.join(project_root, "data/test_data/MATH-500.parquet")
        df = pd.read_parquet(path)
        print(df.head())
        print(df.columns)
        ds = load_dataset(category, split=split)
        for ex in ds:
            results.append({"problem": ex["problem"], "answer": ex["answer"], "subject": "math500"})
    if max_samples > 0:
        results = results[:max_samples]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=1024*16)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--add_flag", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default='<start>')
    parser.add_argument("--save_path", type=str, default="/scratch/pioneer/jobs/user/expert/no_think_collect")
    args = parser.parse_args()

    save_path = os.path.join(
        args.save_path,
        f"{args.model.split('/')[-1]}_{args.enable_thinking}_{args.dataset.split('/')[-1]}_{args.suffix}.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 检查已生成条数
    existing = 0
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            existing = sum(1 for _ in f)
        print(f"⚠️ 检测到已有 {existing} 条记录，将从第 {existing} 条之后继续生成。")

    rows = get_category_data(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(rows)} samples from {args.dataset}")


    # 初始化 tokenizer 与 vLLM
    print(f"🚀 Loading {args.model} with vLLM ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )

    
    prompts = []
    for ex in rows[existing:]:
        # 用 tokenizer 模板生成标准化的 prompt
        messages = [{"role": "user", "content": ex["problem"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=args.enable_thinking
        )

        # no-think 模式时追加空 think 块 + suffix
        if not args.enable_thinking:
            text += "<think>\n\n</think>\n\n" + args.suffix
        else:
            text += args.suffix
        prompts.append(text)

    outputs = llm.generate(prompts, sampling_params)

    acc = 0
    with open(save_path, "a") as fout:
        for i, (ex, output) in enumerate(zip(rows[existing:], outputs)):
            gen_text = output.outputs[0].text
            correct = True # check_correctness(gen_text, ex["answer"])
            acc += int(correct)
            fout.write(json.dumps({
                "len": len(gen_text),
                "correct": correct,
                "problem": ex["problem"],
                "input_text": prompts[i],
                "generated_text": gen_text,
                "answer": ex["answer"]
            }, ensure_ascii=False) + "\n")
            fout.flush()

    total = len(rows) - existing
    print(f"✅ 生成完成: {save_path}, 准确率 {acc/total:.4f} ({acc}/{total})")


if __name__ == "__main__":
    main()