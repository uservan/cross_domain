python -m verl.model_merger merge --backend fsdp \
    --local_dir /scratch/pioneer/jobs/user/verl/model/MIX-REWARD-GRPO-SMALL-Normal/vanilla-Qwen3-4B-Base-math-sl/global_step_234/actor \
    --target_dir /scratch/pioneer/jobs/user/verl/model/MIX-REWARD-GRPO-SMALL-Normal-merge3/Qwen3-4B-Base-math-sl \
    --private