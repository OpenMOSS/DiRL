# 分布式环境变量（兼容 torchrun / accelerate / Slurm 等）
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}  # 注意：有些系统用 RANK 表示 node rank

# 自动推导 num_machines（节点数）
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_MACHINES=${WORLD_SIZE}
TOTAL_GPUS=$((NUM_MACHINES * NUM_GPUS_PER_NODE))

# 设置 CUDA 相关环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 设置每个进程独立的编译缓存目录，避免多进程写冲突
export TRITON_CACHE_DIR=/tmp/triton_cache_rank${RANK:-0}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_rank${RANK:-0}

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=./cache/${SCRIPT_NAME}

# 清理冲突变量
unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING

# RL 参数（可通过环境变量传入）
SCRIPT_NAME="${SCRIPT_NAME:-rl_sdar}"
RUN_NAME=WANDB_${SCRIPT_NAME}
export WANDB_PROJECT=${RUN_NAME}
export WANDB_DIR=${WANDB_PROJECT}
mkdir -p "$WANDB_DIR"
export WANDB_MODE=offline

PRETRAINED_MODEL=${PRETRAINED_MODEL:-xxx/SDAR-8B-Chat}
NUM_TASK_PER_STEP=${NUM_TASK_PER_STEP:-128}
NUM_RESPONSE_PER_TASK=${NUM_RESPONSE_PER_TASK:-8}
REWARD_FUNCS=${REWARD_FUNCS:-accuracy}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
BLOCK_SIZE=${BLOCK_SIZE:-4}
MAX_TOKEN=${MAX_TOKEN:-8192}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-1.0}
TEMPERATURE=${TEMPERATURE:-1.0}
DENOISING_STEPS_PER_BLOCK=${DENOISING_STEPS_PER_BLOCK:-4}
REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
DYNAMIC_THRESHOLD=${DYNAMIC_THRESHOLD:-0.90}
SAVE_EVERY=${SAVE_EVERY:-1}
TRAIN_DATASET=${TRAIN_DATASET:-MATH_train}
EVAL_DATASET=${EVAL_DATASET:-MATH500}
EVAL_BLOCK_SIZE=${EVAL_BLOCK_SIZE:-4}
EVAL_DENOISING_STEPS=${EVAL_DENOISING_STEPS:-4}
EVAL_MAX_TOKEN=${EVAL_MAX_TOKEN:-8192}
EVAL_TOP_K=${EVAL_TOP_K:-50}
EVAL_TOP_P=${EVAL_TOP_P:-1.0}
EVAL_REMASKING_STRATEGY=${EVAL_REMASKING_STRATEGY:-low_confidence_dynamic}
EVAL_DYNAMIC_THRESHOLD=${EVAL_DYNAMIC_THRESHOLD:-0.90}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
EVAL_EVERY=${EVAL_EVERY:-0}
KL=${KL:-0}
SHRINK=${SHRINK:-1}
THINK=${THINK:-False}
LR=${LR:-1e-6}
BS=${BS:-1}
MINI_BS=${MINI_BS:-0}
CUSOR=${CUSOR:-0}
ITERATIONS=${ITERATIONS:-1}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-10}
CURRENT_STEP=${CURRENT_STEP:-1}
echo "[NUM NODES: $NUM_MACHINES] RL training start ..."

# # 显存优化：减少碎片化
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # --config_file accelerate_configs/${WORLD_SIZE}_node_8_gpus_deepspeed_zero1.yaml \
# # 启动训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --num_machines ${NUM_MACHINES} \
  --config_file accelerate_configs/deepspeed_zero1.yaml \
  --num_processes ${TOTAL_GPUS} \
  train/dapo.py \
  config=configs/rl.yaml \
  model.pretrained_model=${PRETRAINED_MODEL} \
  experiment.project=${SCRIPT_NAME}_lmdeploy \
  experiment.cursor=${CUSOR} \
  experiment.num_nodes=${NUM_MACHINES} \
  experiment.save_every=${SAVE_EVERY} \
  dataset.train_dataset=${TRAIN_DATASET} \
  dataset.eval_dataset=${EVAL_DATASET} \
  models.pretrained_model=${PRETRAINED_MODEL} \
  experiment.current_step=${CURRENT_STEP} \
  training.num_train_steps=${NUM_TRAIN_STEPS} \
  training.block_size=${BLOCK_SIZE} \
  training.batch_size_lm=${BS} \
  training.mini_batch_size=${MINI_BS} \
  training.num_iterations=${ITERATIONS} \
  rollout.start_with_think=${THINK} \
  rollout.num_task_per_step=${NUM_TASK_PER_STEP} \
  rollout.num_response_per_task=${NUM_RESPONSE_PER_TASK} \
  training.beta=${KL} \
  training.shrink=${SHRINK} \
  training.reward_funcs=${REWARD_FUNCS} \
  optimizer.params.learning_rate=${LR} \
  training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  rollout.block_size=${BLOCK_SIZE} \
  rollout.max_token=${MAX_TOKEN} \
  rollout.top_k=${TOP_K} \
  rollout.top_p=${TOP_P} \
  rollout.temperature=${TEMPERATURE} \
  rollout.denoising_steps_per_block=${DENOISING_STEPS_PER_BLOCK} \
  rollout.remasking_strategy=${REMASKING_STRATEGY} \
  rollout.dynamic_threshold=${DYNAMIC_THRESHOLD} \
  rollout.do_sample=True \
  evaluation.do_sample=True \
  evaluation.block_size=${EVAL_BLOCK_SIZE} \
  evaluation.denoising_steps_per_block=${EVAL_DENOISING_STEPS} \
  evaluation.max_active=${EVAL_MAX_ACTIVE} \
  evaluation.max_token=${EVAL_MAX_TOKEN} \
  evaluation.top_k=${EVAL_TOP_K} \
  evaluation.remasking_strategy=${EVAL_REMASKING_STRATEGY} \
  evaluation.dynamic_threshold=${EVAL_DYNAMIC_THRESHOLD} \
  evaluation.temperature=${EVAL_TEMPERATURE} \
  experiment.eval_every=${EVAL_EVERY}

echo "RL training finished ..."

