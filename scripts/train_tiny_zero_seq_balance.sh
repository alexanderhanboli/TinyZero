set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

export BASE_MODEL=Qwen/Qwen2.5-3B
export ROLLOUT_TP_SIZE=2
export PROJECT_NAME=TinyZero-mixed
export EXPERIMENT_NAME=dynamic-batching-qwen-2.5-3b
export HYDRA_FULL_ERROR=1

export WANDB_API_KEY=be7f64dd8438dcb43e912a32be795ebc65455162

math_train_path=$HOME/efs/tinyzero/math-300/train.parquet
math_test_path=$HOME/efs/tinyzero/math-300/test.parquet
qrtext_train_path=$HOME/efs/tinyzero/qrtext/train.parquet
qrtext_test_path=$HOME/efs/tinyzero/qrtext/test.parquet

train_files="['$math_train_path', '$qrtext_train_path']"
test_files="['$qrtext_test_path']"

# Generate a GPU-based identifier
GPU_IDENTIFIER="gpus-${CUDA_VISIBLE_DEVICES//,/}"

# Set dynamic log filename
LOG_FILENAME="verl_${PROJECT_NAME}_${EXPERIMENT_NAME}_${GPU_IDENTIFIER}.log"

# Set dynamic checkpoint directory
CKPT_DIR="$HOME/efs/tinyzero/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}_${GPU_IDENTIFIER}"

max_token_len_per_gpu=4000

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node=$N_GPUS \
    +trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee "$LOG_FILENAME"