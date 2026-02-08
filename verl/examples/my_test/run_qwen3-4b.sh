#!/bin/bash
#SBATCH --job-name=test
#SBATCH -A aiscii
#SBATCH --partition=aiscii
#SBATCH --gres=gpu:4
#SBATCH -n 32
#SBATCH --mem=128G
#SBATCH --output=/scratch/pioneer/jobs/wxy320/verl/slurm/qwen_rope_%A_%a.txt
#SBATCH --error=/scratch/pioneer/jobs/wxy320/verl/slurm/qwen_rope_%A_%a.txt
#SBATCH --time=13-00:00:00

# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED=0

save_contents="['hf_model']"

# 先定义数据目录
DATA_DIR=/scratch/pioneer/jobs/wxy320/verl/data
    # data.train_files=$DATA_DIR/train.parquet \
    # data.val_files=$DATA_DIR/test.parquet \

gsm8k_test_path=data/test_data/gsm8k_test.parquet
math_test_path=data/test_data/math500.parquet
amc23_test_path=data/test_data/amc23.parquet
aime24_test_path=data/test_data/aime24.parquet
aime25_test_path=data/test_data/aime25.parquet
minerva_test_path=data/test_data/minerva_math.parquet
olympiadbench_test_path=data/test_data/olympiadbench.parquet

math_train_path=data/train/MATH.parquet
dapo_math_train_path=data/train/DAPO-Math-17k.parquet

train_files="['$math_train_path']"
test_files="['$aime24_test_path', '$aime25_test_path']"

model_path=Qwen/Qwen3-1.7B
# Number of CPUs for Ray. Use a fixed number instead of null when using SLURM.
project_name="verl_grpo_example_gsm8k_math"
experiment_name="qwen3_1.7b_math_grpo"

nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=16 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir=${DATA_DIR}/rollout_data/$project_name/$experiment_name \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=400 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    > /scratch/pioneer/jobs/wxy320/verl/log/my_test/${experiment_name}.log 2>&1