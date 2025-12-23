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

# 清理冲突变量
unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING

MODEL="${MODEL:-sdar}" 
DATASET_TYPE="${DATASET_TYPE:-math}"
RUN_NAME=WANDB_${MODEL}_eval_${DATASET_TYPE}
export WANDB_PROJECT=$RUN_NAME
export WANDB_DIR=${WANDB_PROJECT}
mkdir -p "$WANDB_DIR"
export WANDB_MODE=offline

# Eval 参数（可通过环境变量传入）
BLOCK_SIZE=${BLOCK_SIZE:-4}
DENOISING_STEPS=${DENOISING_STEPS:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_TOKEN=${MAX_TOKEN:-8192}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-1.0}
NUM_RESPONSE=${NUM_RESPONSE:-16}
EVAL_DATASET=${EVAL_DATASET:-MATH500}
MODEL_PATH=${MODEL_PATH:-xxx/SDAR-8B-Chat}
REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
DYNAMIC_THRESHOLD=${DYNAMIC_THRESHOLD:-0.9}
DO_SAMPLE=${DO_SAMPLE:-True}
THINK=${THINK:-False}

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=xxx/cache/${SCRIPT_NAME}


echo "[NUM NODES: $NUM_MACHINES] Eval start ..."

# # 启动sample
accelerate launch \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --num_machines ${NUM_MACHINES} \
  --num_processes ${TOTAL_GPUS} \
  --config_file accelerate_configs/multi_gpus.yaml \
  eval/eval_lmdeploy.py \
  config=configs/eval.yaml \
  experiment.num_nodes=${NUM_MACHINES} \
  rollout.start_with_think=${THINK} \
  rollout.block_size=${BLOCK_SIZE} \
  rollout.denoising_steps_per_block=${DENOISING_STEPS} \
  rollout.temperature=${TEMPERATURE} \
  rollout.top_k=${TOP_K} \
  rollout.top_p=${TOP_P} \
  rollout.max_token=${MAX_TOKEN} \
  rollout.num_response_per_task=${NUM_RESPONSE} \
  dataset.data_type=${DATASET_TYPE} \
  model=${MODEL_PATH} \
  dataset.eval_dataset=${EVAL_DATASET} \
  rollout.remasking_strategy=${REMASKING_STRATEGY} \
  rollout.dynamic_threshold=${DYNAMIC_THRESHOLD} \
  rollout.do_sample=${DO_SAMPLE}
