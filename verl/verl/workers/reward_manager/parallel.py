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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import ray
@ray.remote
def _compute_wrapper(job, compute_fn):
    try:
        return compute_fn(**job)
    except Exception as e:
        return {"score": -1, "acc": False, "preds": f"error: {e}"}

@register("parallel")
class ParallelRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
    
    def _parallel_compute_scores(self, jobs, max_workers=64, timeout_s=10, desc="compute_score"):
        results = [{"score": -1, "acc": False, "preds": "timeout", "data_source":"None"} for _ in range(len(jobs))]
        # iterator = tqdm(total=len(jobs), desc=desc)

        futures = {}
        for idx, job in enumerate(jobs):
            fut = _compute_wrapper.remote(job, self.compute_score)
            futures[fut] = idx

            if len(futures) >= max_workers:
                done, _ = ray.wait(list(futures.keys()), num_returns=1, timeout=timeout_s)
                if not done:
                    print(f"⚠️ No task finished in {timeout_s}s, retrying...")
                    continue
                finished = done[0]
                id = futures.pop(finished)
                try:
                    results[id] = ray.get(finished, timeout=1)
                except Exception as e:
                    results[id] = {"score": -1, "acc": False, "preds": f"error: {e}", "data_source":"None"}
                # iterator.update(1)

        while futures:
            done, _ = ray.wait(list(futures.keys()), num_returns=1, timeout=timeout_s + 30)
            if not done:
                print(f"⚠️ {len(futures)} unfinished tasks exceeded timeout, cancelling them...")
                for fut, idx in list(futures.items()):
                    try:
                        ray.cancel(fut, force=True)  
                    except Exception as e:
                        print(f"⚠️ Cancel failed for task {idx}")
                    results[idx] = {"score": -1, "acc": False, "preds": f"timeout>{timeout_s+30}s", "data_source":"None"}
                    del futures[fut]
                break 

            finished = done[0]
            idx = futures.pop(finished)
            try:
                results[idx] = ray.get(finished, timeout=1)
            except Exception as e:
                results[idx] = {"score": -1, "acc": False, "preds": f"error: {e}", "data_source":"None"}
            # iterator.update(1)
         
        # === Cleanup: prevent OOM ===
        try:
            # ONLY free ray object refs; results is safe and untouched
            ray.internal.free(list(futures.keys()), local_only=False)
        except:
            pass
        # clean python variables
        import gc
        del futures
        gc.collect()

        return results

    def _parallel_compute_scores_old(self, jobs, max_workers=64, desc="compute_score"):
        """
        Safe parallel version for Ray environment.
        compute_score: performs lightweight computation / math reward comparison.
        """
        results = [None] * len(jobs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.compute_score, **job): idx for idx, job in enumerate(jobs)}

            with tqdm(total=len(jobs), desc=desc) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result(timeout=10)
                    except Exception as e:
                        # 记录异常，防止单个任务阻塞所有结果
                        results[idx] = {"score": -1, "acc": False, "preds": f"error: {e}", "data_source":"None"}
                    pbar.update(1)

        return results

    def __call__(self, data: DataProto, return_dict: bool = False, parallel=True, max_workers=64):
        if "rm_scores" in data.batch.keys():
            ...
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # === Step 1: extract all compute_score jobs ===
        jobs = []
        decode_cache = []  # keep decoded data for later assembly

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            jobs.append(dict(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ))

            decode_cache.append((i, valid_response_length, data_source, prompt_str, response_str, ground_truth))

        # === Step 2: parallel compute or fallback to sequential ===
        if parallel:
            all_results = self._parallel_compute_scores(jobs, max_workers=max_workers)
        else:
            # original sequential version
            all_results = [self.compute_score(**job) for job in jobs]

        # === Step 3: assemble rewards ===
        already_print_data_sources = {}
        for (i, valid_response_length, data_source, prompt_str, response_str, ground_truth), result in zip(decode_cache, all_results):
            if isinstance(result, dict):
                score = result["score"]
                for key, val in result.items():
                    reward_extra_info[key].append(val)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score

            # apply overlong penalty
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            # 👀 print debug sample
            if already_print_data_sources.get(data_source, 0) < self.num_examine:
                already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0) + 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score_dict]" if isinstance(result, dict) else "[score]", result)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
