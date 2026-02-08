import re
from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed
from typing import Dict, Optional, Tuple

def get_multiple_choice_answer(pred: str):
    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred

def check_correctness(self, problem, generation):
    pred = get_multiple_choice_answer(generation)
    answer = problem[self.task_config.answer_key]
    return answer == pred

def extract_single_letter(text: str):
    matches = re.findall(r"\b[A-Za-z]\b", text)
    return matches[0].upper() if matches else None

def compute_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(solution_str)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else '[INVALID]'
    # boxed_pred = last_boxed_only_string(solution_str)
    # if boxed_pred is not None:
    #     extracted_pred = remove_boxed(boxed_pred) 
    #     extracted_pred = extract_single_letter(extracted_pred) 
    #     if extracted_pred: extracted_pred = '[INVALID]'
    # else:
    #     extracted_pred = '[INVALID]'

    acc = extracted_pred.strip().lower() == ground_truth.lower()
    reward = 1.0 if acc else -1.0

    return {
        "score": reward,
        "acc": acc,
        "preds": extracted_pred,
    }