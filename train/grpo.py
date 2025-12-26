import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
import contextlib
from pathlib import Path
from openai import OpenAI
from typing import Union
import requests
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
import socket
from transformers import AutoTokenizer
from accelerate import Accelerator
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig, serve
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out

from models import SDARForCausalLM
from train.prompting_utils import UniversalPrompting
from train.server import get_host_ip, start_server, server_sleep, server_update_weights, server_wakeup


from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader
from train.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

from rollout_dapo_server import rollout_sampling, rollout_sampling_eval

logger = get_logger(__name__, log_level="INFO")


# ============= Dynamic Threshold Scheduler =============
def get_dynamic_confidence_threshold(current_step, total_steps,
                                     start_threshold=0.75,
                                     end_threshold=0.95):
    """
    动态置信度阈值：从低到高线性增长
    
    初期模型弱 -> 阈值低(0.75) -> 容易unmask
    后期模型强 -> 阈值高(0.95) -> 抑制过快解码
    """
    progress = min(1.0, current_step / max(total_steps, 1))
    return start_threshold + (end_threshold - start_threshold) * progress
# ========================================================

import torch.distributed as dist
from transformers import AutoTokenizer

import os


class TrainDataset(Dataset):
    def __init__(self, input_ids, p_mask, masked_indices, valid_indices, labels, rewards, raw_rewards, is_real, step_ids, ref_logprobs=None, old_logprobs=None):
        self.input_ids = input_ids
        self.p_mask = p_mask
        self.masked_indices = masked_indices
        self.valid_indices = valid_indices
        self.labels = labels
        self.rewards = rewards
        self.raw_rewards = raw_rewards
        self.is_real = is_real  # 标记是否为真实样本
        self.step_ids = step_ids  # 标记每个token属于哪一步（用于联合概率计算）
        self.ref_logprobs = ref_logprobs  # 参考模型的logprob（预先计算好的）
        self.old_logprobs = old_logprobs  # old policy的logprob（GRPO用）

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.ref_logprobs is not None:
            ref_logprob = self.ref_logprobs[idx]
        else:
            # Return an empty tensor instead of None to avoid collate errors
            ref_logprob = torch.tensor([])
        
        if self.old_logprobs is not None:
            old_logprob = self.old_logprobs[idx]
        else:
            old_logprob = torch.tensor([])
        
        return (
            self.input_ids[idx],
            self.p_mask[idx],
            self.masked_indices[idx],
            self.valid_indices[idx],
            self.labels[idx],
            self.rewards[idx],
            self.raw_rewards[idx],
            self.is_real[idx],
            self.step_ids[idx],
            ref_logprob,
            old_logprob,
        )


def train():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    if config.experiment.current_step == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = config.experiment.prefix_dir + project_name + f"/ckpt/step-{config.experiment.current_step-1}"
    model_name = "optimized_model"

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    
    # 先用默认值初始化 Accelerator（为了获取 num_processes）
    temp_accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )
    
    # 现在可以获取 num_processes 了
    num_processes = temp_accelerator.num_processes
    print(f"=== Got num_processes from Accelerator: {num_processes} ===")
    
    # 从配置中获取参数
    num_task_per_step = config.rollout.num_task_per_step
    num_response_per_task = config.rollout.num_response_per_task
    batch_size_train = config.training.batch_size_lm
    block_size = config.training.block_size
    shrink = config.training.get("shrink", 1)
    mini_batch_size = config.training.get("mini_batch_size", 0)
    if mini_batch_size == 0:
        mini_batch_size = num_task_per_step
    
    # 计算扩展倍数：(block_size / shrink) / batch_size_lm
    expansion_factor = (block_size / shrink) / batch_size_train
    
    # 计算有效的梯度累积步数（考虑数据扩展）
    # = (mini_batch_size * num_response_per_task / num_processes) * expansion_factor
    gradient_accumulation_steps = int((mini_batch_size * num_response_per_task / num_processes) * expansion_factor)
    
    print(f"[Pre-init] Calculated gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  - Num processes: {num_processes}")
    print(f"  - Mini batch size: {mini_batch_size}")
    print(f"  - Num response per task: {num_response_per_task}")
    print(f"  - Batch size train: {batch_size_train}")
    print(f"  - Block size: {block_size}")
    print(f"  - Shrink: {shrink}")
    print(f"  - Expansion factor: {expansion_factor}")
    
    # 覆盖配置中的梯度累积步数
    config.training.gradient_accumulation_steps = gradient_accumulation_steps
    
    # 删除临时的 accelerator
    del temp_accelerator
    
    # 用正确的梯度累积步数重新初始化 Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # 手动控制累积，不使用accelerator的自动累积
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
        # kwargs_handlers=[ddp_kwargs],
    )
    
    # 验证Accelerator设置
    if accelerator.is_main_process:
        print(f"[Post-init] Accelerator.gradient_accumulation_steps: {accelerator.gradient_accumulation_steps}")
        print(f"[Post-init] Manual gradient_accumulation_steps: {gradient_accumulation_steps}")

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize wandb tracking (on all processes)
    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_init_kwargs = dict(
        name=config.experiment.project,
        id=run_id,
        resume=resume_wandb_run,
        entity=config.wandb.get("entity", None),
        config_exclude_keys=[],
    )
    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
    wandb_config.pop("experiment.resume_from_checkpoint", None)

    accelerator.init_trackers(
        config.experiment.project,
        config=wandb_config,
        init_kwargs={"wandb": wandb_init_kwargs},
    )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    # Initialize global step counter
    global_step = 0  # counts every batch
    optimizer_step = 0  # counts every optimizer update (after gradient accumulation)
    model = None
    ref_model = None  # Reference model for KL divergence
    optimizer = None
    lr_scheduler = None
    # 数据集遍历位置，用于顺序遍历（支持续训）
    data_cursor = getattr(config.experiment, 'cursor', 0)
    if data_cursor > 0:
        logger.info(f"Resuming from data cursor: {data_cursor}")

    import torch.nn.functional as F

    local_ip = get_host_ip()
    rank = accelerator.process_index
    port = 12333 + 10*int(rank)

    backend_config = PytorchEngineConfig(
        dtype="auto",  # 自动从模型config读取，与训练保持一致
        cache_max_entry_count=0.9,
        dllm_block_length=config.rollout.block_size,
        dllm_denoising_steps=config.rollout.denoising_steps_per_block,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
        max_prefill_token_num=config.rollout.max_token * 64,
    )
    client = start_server(local_ip, rank, port, pretrained_model, model_name, backend_config)
    accelerator.wait_for_everyone()
    base_url = f'http://{local_ip}:{port}'

    for step in range(config.experiment.current_step, config.training.num_train_steps+1):

        logger.info(f"{'='*80}")
        logger.info(f"Starting Step {step}/{config.training.num_train_steps}")
        logger.info(f"{'='*80}")
        
        # ============ 时间统计 ============
        time_stats = {}
        step_start_time = time.time()

        #########################
        # 每轮的rollout采样  #
        #########################

        if step != config.experiment.current_step or config.experiment.current_step != 1:
            model_path = config.experiment.prefix_dir + project_name + f"/ckpt/step-{step-1}"
        else:
            model_path = pretrained_model
        logger.info(f"Loading model from {model_path}")
        
        # tokenizer 应该从训练开始前的初始路径加载，防止上个epoch没有保存
        tokenizer_path = config.model.pretrained_model
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        # 防止加载导入时冲突
        sys.path.insert(0, tokenizer_path)
        from tokenization_qwen2 import Qwen2Tokenizer
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)

        logger.info(f"Starting rollout sampling for step {step}...")
        
        # ========== 开始 Rollout 阶段计时 ==========
        rollout_start_time = time.time()

        if model is not None:
            model = model.cpu()

        accelerator.wait_for_everyone()

        # ========== 动态阈值调度 ==========
        # 计算总训练步数用于阈值调度
        num_train_steps = config.training.num_train_steps
        num_iterations = config.training.get("num_iterations", 8)
        total_train_iters = num_iterations * num_train_steps
        
        # 计算当前动态阈值
        # current_dynamic_threshold = get_dynamic_confidence_threshold(
        #     current_step=step-1,
        #     total_steps=num_train_steps,
        #     start_threshold=0.95,  # 初期低阈值
        #     end_threshold=0.99     # 后期高阈值
        # )
        current_dynamic_threshold = config.rollout.dynamic_threshold
        
        if accelerator.is_main_process:
            logger.info(f"Dynamic Threshold: {current_dynamic_threshold:.3f} (step {step}/{num_train_steps})")
        
        # Wakeup server before rollout sampling（确保 weights 在 GPU 上）
        # logger.info(f"[Rank {accelerator.process_index}] Waking up server for rollout sampling")
        # server_wakeup(base_url)
        
        # 注意：现在 rollout_sampling 返回 (data, updated_cursor, time_breakdown) 元组
        dataset_load, data_cursor, rollout_time_breakdown = rollout_sampling(
            client,
            config.dataset.train_dataset, step, config,
            model_path, model_name, tokenizer, accelerator,
            normalize=True, filter=True, mode="train", split_rank=False, reward_funcs=config.training.reward_funcs,
            data_cursor=data_cursor,  # 传入当前cursor，顺序遍历数据集
            dynamic_threshold=current_dynamic_threshold  # 传入动态阈值
        )
        
        # Sleep server after rollout sampling
        logger.info(f"[Rank {accelerator.process_index}] Putting server to sleep after rollout sampling")

        logger.info(f"[Rank {accelerator.process_index}] Rollout sampling completed. Collected {len(dataset_load)} samples.")

        sleep_start_time = time.time()
        server_sleep(base_url, level=2)
        sleep_time = time.time() - sleep_start_time
        time_stats['sleep'] = sleep_time
        print(f"[Time Stats] Sleep: {sleep_time:.2f}s")
        
        accelerator.wait_for_everyone()
        
        # ========== 结束 Rollout 阶段计时 ==========
        rollout_time = time.time() - rollout_start_time
        time_stats['rollout'] = rollout_time
        
        # 统计本地token数
        local_tokens = 0
        for item in dataset_load:
            if isinstance(item, dict):
                token_lengths = item.get("token_length", [])
                local_tokens += sum(token_lengths)
        
        # Gather所有rank的统计
        from accelerate.utils import gather_object
        all_rollout_data = gather_object([{
            'tokens': local_tokens,
            'rollout_time': rollout_time
        }])
        
        # 计算集群总TPS
        global_tokens = sum(d['tokens'] for d in all_rollout_data)
        max_rollout_time = max(d['rollout_time'] for d in all_rollout_data)  # 最慢的GPU决定总时间
        cluster_tps = global_tokens / max_rollout_time if max_rollout_time > 0 else 0.0
        
        time_stats['rollout_tokens'] = global_tokens
        time_stats['sampling_throughput'] = cluster_tps
        
        logger.info(f"[Time Stats] Rollout phase completed in {rollout_time:.2f}s")

        accelerator.wait_for_everyone()
        
        # rollout 完成后，将模型和optimizer移回 GPU 继续训练
        if model is not None:
            logger.info(f"[Rank {accelerator.process_index}] Moving model back to GPU for training")
            model.to(accelerator.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 将optimizer状态移回GPU (已取消此操作)
        # if optimizer is not None:
        #     opt_to_gpu_start = time.time()
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.to(accelerator.device)
        #     opt_to_gpu_time = time.time() - opt_to_gpu_start
        #     logger.info(f"[Rank {accelerator.process_index}] Moved optimizer state back to GPU in {opt_to_gpu_time:.2f}s")
        
        accelerator.wait_for_everyone()
        
        # 检查显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"[Rank {accelerator.process_index}] After server_sleep - GPU Memory: "
                       f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max Allocated: {max_allocated:.2f}GB")
        # time.sleep(20)
        # Log token length statistics from rollout
        try:
            token_lengths = []
            correctness_list = []
            raw_rewards_list = []
            reward_components_list = []
            extract_rewards_list = []
            speed_metrics_list = []
            for item in dataset_load:
                if isinstance(item, dict):
                    if "token_length" in item:
                        token_lengths.extend(item["token_length"])
                    if "correctness" in item:
                        correctness_list.extend(item["correctness"])
                    if "raw_rewards" in item:
                        raw_rewards_list.extend(item["raw_rewards"])
                    if "reward_components" in item:
                        reward_components_list.extend(item["reward_components"])
                    if "extract_reward" in item:
                        extract_rewards_list.extend(item["extract_reward"])
                    if "speed_metrics" in item:
                        speed_metrics_list.extend(item["speed_metrics"])
            
            # Gather 所有 rank 的数据（一次性打包）
            from accelerate.utils import gather_object
            local_data = {
                'num_tasks': len(dataset_load),
                'token_lengths': token_lengths,
                'correctness': correctness_list,
                'raw_rewards': raw_rewards_list,
                'reward_components': reward_components_list,
                'extract_rewards': extract_rewards_list,
                'speed_metrics': speed_metrics_list,
            }
            all_data = gather_object([local_data])
            
            # 解包并合并所有 rank 的数据
            all_num_tasks = sum(d['num_tasks'] for d in all_data)
            all_token_lengths = [item for d in all_data for item in d['token_lengths']]
            all_correctness = [item for d in all_data for item in d['correctness']]
            all_raw_rewards = [item for d in all_data for item in d['raw_rewards']]
            all_reward_components = [item for d in all_data for item in d['reward_components']]
            all_extract_rewards = [item for d in all_data for item in d['extract_rewards']]
            all_speed_metrics = [item for d in all_data for item in d['speed_metrics']]
            
            metrics_rollout = {
                "rollout/num_tasks": float(all_num_tasks),
                "rollout/step": float(step),
            }
            
            if len(all_token_lengths) > 0:
                token_arr = np.asarray(all_token_lengths, dtype=np.int64)
                max_token = int(getattr(config.rollout, "max_token", 0) or 0)
                trunc_rate = float((token_arr >= max_token).mean()) if max_token > 0 else 0.0
                metrics_rollout.update({
                    "rollout/num_responses": float(len(all_token_lengths)),
                    "rollout/token_length_mean": float(token_arr.mean()),
                    "rollout/token_length_p50": float(np.percentile(token_arr, 50)),
                    "rollout/token_length_p95": float(np.percentile(token_arr, 95)),
                    "rollout/token_length_min": float(token_arr.min()),
                    "rollout/token_length_max": float(token_arr.max()),
                    "rollout/token_trunc_rate": trunc_rate,
                })
            
            if len(all_correctness) > 0:
                corr_arr = np.asarray(all_correctness, dtype=np.float32)
                metrics_rollout["rollout/correctness"] = float(corr_arr.mean())
            
            if len(all_speed_metrics) > 0:
                speed_arr = np.asarray(all_speed_metrics, dtype=np.float32)
                metrics_rollout["rollout/speed_metric"] = float(speed_arr.mean())
            
            if len(all_extract_rewards) > 0:
                ext_rew_arr = np.asarray(all_extract_rewards, dtype=np.float32)
                metrics_rollout["rollout/extract_reward"] = float(ext_rew_arr.mean())
            
            # 统计raw_reward（加权求和后的值）
            if len(all_raw_rewards) > 0:
                raw_rew_arr = np.array(all_raw_rewards, dtype=np.float32)
                metrics_rollout["rollout/raw_reward"] = float(raw_rew_arr.mean())
            
            # 统计每个reward function的均值
            if len(all_reward_components) > 0:
                # all_reward_components是列表的列表，每个元素是一个列表[reward1, reward2, ...]
                reward_func_names = config.training.get("reward_funcs", "accuracy").split(',')
                reward_func_names = [name.strip() for name in reward_func_names]
                
                # 获取 reward_weights
                reward_weights_str = config.training.get("reward_weights", None)
                if reward_weights_str is not None:
                    if isinstance(reward_weights_str, str):
                        reward_weights = [float(w.strip()) for w in reward_weights_str.split(',') if w.strip()]
                    else:
                        reward_weights = list(reward_weights_str)
                else:
                    # 默认权重：所有reward function权重为1
                    reward_weights = [1.0] * len(reward_func_names)
                
                # 转换为numpy数组: shape = (num_samples, num_reward_funcs)
                comp_matrix = np.array(all_reward_components, dtype=np.float32)
                
                # 如果是1D数组（兼容性处理），reshape为2D
                if comp_matrix.ndim == 1:
                    comp_matrix = comp_matrix.reshape(-1, 1)
                
                # 计算每个reward function的均值
                for idx, func_name in enumerate(reward_func_names):
                    if idx < comp_matrix.shape[1]:
                        func_rewards = comp_matrix[:, idx]
                        metrics_rollout[f"rollout/reward_{func_name}"] = float(func_rewards.mean())
            
            if accelerator.is_main_process:
                # 打印基本统计信息
                logger.info(f"\n{'='*60}")
                logger.info(f"Rollout Statistics - Step {step}")
                logger.info(f"{'='*60}")
                logger.info(f"Total tasks collected: {metrics_rollout['rollout/num_tasks']:.0f}")
                
                if "rollout/num_responses" in metrics_rollout:
                    logger.info(f"Token length stats:")
                    logger.info(f"  Mean: {metrics_rollout['rollout/token_length_mean']:.1f}")
                    logger.info(f"  Min: {metrics_rollout['rollout/token_length_min']:.1f}")
                    logger.info(f"  Median (p50): {metrics_rollout['rollout/token_length_p50']:.1f}")
                    logger.info(f"  p95: {metrics_rollout['rollout/token_length_p95']:.1f}")
                    logger.info(f"  Max: {metrics_rollout['rollout/token_length_max']:.1f}")
                    logger.info(f"  Truncation rate: {metrics_rollout['rollout/token_trunc_rate']:.3f}")
                
                if "rollout/correctness" in metrics_rollout:
                    logger.info(f"Correctness: {metrics_rollout['rollout/correctness']:.4f}")
                
                if "rollout/speed_metric" in metrics_rollout:
                    logger.info(f"Speed metric (avg tokens/step): {metrics_rollout['rollout/speed_metric']:.2f}")
                
                if "rollout/extract_reward" in metrics_rollout:
                    logger.info(f"Extract reward: {metrics_rollout['rollout/extract_reward']:.4f}")
                
                if "rollout/raw_reward" in metrics_rollout:
                    logger.info(f"Total raw reward (weighted sum): {metrics_rollout['rollout/raw_reward']:.4f}")
                
                # 打印每个reward function的均值
                reward_func_names = config.training.get("reward_funcs", "accuracy").split(',')
                reward_func_names = [name.strip() for name in reward_func_names]
                
                # 获取 reward_weights 用于显示
                reward_weights_str = config.training.get("reward_weights", None)
                if reward_weights_str is not None:
                    if isinstance(reward_weights_str, str):
                        reward_weights = [float(w.strip()) for w in reward_weights_str.split(',') if w.strip()]
                    else:
                        reward_weights = list(reward_weights_str)
                else:
                    reward_weights = [1.0] * len(reward_func_names)
                
                logger.info(f"Individual reward functions:")
                for idx, func_name in enumerate(reward_func_names):
                    key = f"rollout/reward_{func_name}"
                    if key in metrics_rollout:
                        weight = reward_weights[idx] if idx < len(reward_weights) else 1.0
                        logger.info(f"  {func_name} (weight={weight}): {metrics_rollout[key]:.4f}")
                
                logger.info(f"{'='*60}\n")
            
            # Rollout metrics是每个step执行一次，使用step作为x轴
            accelerator.log(metrics_rollout, step=step)

        except Exception as e:
            logger.warning(f"Failed to compute/log rollout stats: {e}")

        config.training.max_gen_length = config.rollout.max_token

        # 训练种子要设置在采样种子后面，防止每个step随机抽取相同的子集题目
        if config.training.seed is not None:
            set_seed(config.training.seed + step)

        uni_prompting = UniversalPrompting(
            tokenizer,
            max_prompt_len=config.training.max_prompt_len,
            max_gen_length=config.training.max_gen_length,
            block_size=config.training.block_size,
            ignore_id=-100
        )

        # 使用 AutoModelForCausalLM 自动根据 config.json 识别模型类型（dense 或 MoE）
        from transformers import AutoModelForCausalLM
        
        # 先不加载训练模型，等ref_logprob计算完再加载

        mask_id = tokenizer.mask_token_id
        pad_id = tokenizer.pad_token_id

        ##################################
        #         内部函数（全部保留）    #
        #################################

        def collapse_k_unique(lst, k: int):
            if k <= 0:
                raise ValueError("k must be > 0")
            uniq = sorted(set(lst))

            mapping = {}
            n = len(uniq)
            for idx, val in enumerate(uniq):
                group = idx // k
                end_idx = min((group + 1) * k - 1, n - 1)
                rep = uniq[end_idx]
                mapping[val] = rep
            return [mapping[x] for x in lst]

        def generate_training_samples_for_one(input_ids_b, step_map_b, labels_b, block_size):
            """
            为单个样本生成所有训练数据
            
            input_ids_b: (L,) 完整序列
            step_map_b: (L,) 完整序列的step_map（-1为无效位置，>0为有效step）
            labels_b: (L,) prompt部分是-100，response部分是token_id
            
            返回: [(p_mask, masked_indices), ...]
            """
            device = input_ids_b.device
            L = input_ids_b.shape[0]
            
            # 只处理response部分（labels != -100）
            response_mask = labels_b != -100
            if not response_mask.any():
                return []
            
            # 按block划分
            NB = (L + block_size - 1) // block_size
            step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
            step_pad[:L] = step_map_b.clone()
            step_blk = step_pad.view(NB, block_size)  # (NB, block_size)
            
            results = []
            big = torch.iinfo(step_blk.dtype).max
            
            # 循环生成训练数据：每轮在每个block中找最小step
            for _ in range(block_size):  # 最多block_size轮
                # 找到每个block当前的最小step（>0）
                valid = step_blk.gt(0)
                if not valid.any():
                    break
                
                tmp = step_blk.masked_fill(~valid, big)
                min_vals, _ = tmp.min(dim=1, keepdim=True)  # (NB, 1)
                
                # 检查是否还有有效的最小值
                if (min_vals == big).all():
                    break
                
                # p_mask: 在每个block中，step等于该block最小step的位置
                pmask_blk = step_blk.eq(min_vals) & valid & (min_vals != big)
                
                # masked_indices: step >= 该block最小step的位置
                ge_mask_blk = (step_blk >= min_vals) & valid & (min_vals != big)
                
                pmask_flat = pmask_blk.view(-1)[:L]
                ge_mask_flat = ge_mask_blk.view(-1)[:L]
                
                # 先将已处理的位置标记为-1
                step_blk = step_blk.masked_fill(pmask_blk, -1)
                
                p_mask_b = pmask_flat & response_mask
                masked_indices_b = ge_mask_flat & response_mask
                
                # 只有有效位置时才加入结果
                if p_mask_b.sum() > 0:
                    assert (p_mask_b & ~masked_indices_b).sum() == 0, "p_mask should be subset of masked_indices"
                    results.append((p_mask_b, masked_indices_b))
            
            return results

        def collect_training_data(input_ids, labels_lm, valid_indices, step_map_tensor, reward, raw_reward):
            """
            收集所有扩展数据（平铺合并）
            
            返回: (input_ids_batch, p_mask, masked_indices, valid_indices_batch, labels_batch, step_ids_batch, reward_list, raw_reward_list, is_real_list, num_real_samples)
            """
            B, L = input_ids.shape
            block_size = config.training.block_size
            
            assert input_ids.shape == (B, L), f"input_ids shape error: {input_ids.shape}"
            assert labels_lm.shape == (B, L), f"labels_lm shape error: {labels_lm.shape}"
            assert step_map_tensor.shape == (B, L), f"step_map_tensor shape error: {step_map_tensor.shape}"
            
            if accelerator.is_main_process:
                print(f"\n=== Collect Training Data Debug ===")
                print(f"Input: B={B}, L={L}, block_size={block_size}")

            if config.training.method == "DiPO":
                step_map = step_map_tensor
                assert step_map.shape == (B, L)

                # 应用shrink功能：把每个block内所有>0的step设置成该block内的最小正值
                if hasattr(config.training, 'shrink') and config.training.shrink > 0:
                    if accelerator.is_main_process:
                        print(f"shrink: {config.training.shrink}")
                    if config.training.shrink != 1:
                        for b in range(B):
                            # 按照整个序列L划分成固定的block
                            num_blocks = (L + block_size - 1) // block_size
                            for blk_idx in range(num_blocks):
                                blk_start = blk_idx * block_size
                                blk_end = min((blk_idx + 1) * block_size, L)
                                
                                # 找到该block内所有>0的step值
                                block_slice = step_map[b, blk_start:blk_end]
                                positive_mask = block_slice > 0
                                
                                if positive_mask.any():
                                    # 找到最小正值
                                    min_positive = block_slice[positive_mask].min().item()
                                    # 把所有>0的位置都设置成最小正值
                                    step_map[b, blk_start:blk_end] = torch.where(
                                        positive_mask,
                                        torch.tensor(min_positive, dtype=step_map.dtype, device=step_map.device),
                                        block_slice
                                    )

                input_ids_list, pmask_list, masked_indices_list, labels_list, valid_indices_list, step_ids_list, reward_list_, raw_reward_list_, is_real_list_ = [], [], [], [], [], [], [], [], []
                num_real_samples = 0  # 统计真实样本数（不包括padding）
                
                for b in range(B):
                    # 生成该样本的所有训练数据
                    samples = generate_training_samples_for_one(
                        input_ids[b], step_map[b], labels_lm[b], block_size
                    )
                    
                    round_count = len(samples)
                    
                    # 先添加真实样本
                    for p_mask_b, masked_indices_b in samples:
                        input_ids_list.append(input_ids[b])
                        pmask_list.append(p_mask_b)
                        masked_indices_list.append(masked_indices_b)
                        labels_list.append(labels_lm[b].clone())
                        valid_indices_list.append(valid_indices[b])
                        step_ids_list.append(step_map[b])  # 添加 step_map 作为 step_ids
                        reward_list_.append(reward[b])
                        raw_reward_list_.append(raw_reward[b])
                        is_real_list_.append(True)  # 标记为真实样本
                        num_real_samples += 1  # 统计真实样本
                    
                    # 清理samples变量，避免累积
                    del samples
                    
                    # 如果不足block_size，补充虚拟样本（reward=0，对loss无贡献）
                    if config.training.shrink == 1 and round_count < block_size and round_count > 0:
                        padding_count = block_size - round_count
                        # 使用全0/全-100的虚假数据作为padding
                        fake_p_mask = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
                        fake_masked_indices = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
                        fake_labels = torch.full((L,), -100, dtype=labels_lm.dtype, device=input_ids.device)  # 全-100
                        fake_valid_indices = torch.zeros(L, dtype=valid_indices.dtype, device=input_ids.device)  # 全0
                        fake_step_ids = torch.zeros(L, dtype=step_map.dtype, device=input_ids.device)  # 全0
                        for _ in range(padding_count):
                            input_ids_list.append(input_ids[b])
                            pmask_list.append(fake_p_mask)
                            masked_indices_list.append(fake_masked_indices)
                            labels_list.append(fake_labels)
                            valid_indices_list.append(fake_valid_indices)
                            step_ids_list.append(fake_step_ids)  # 添加 fake step_ids
                            reward_list_.append(0.0)  # reward=0
                            raw_reward_list_.append(0.0)  # 使用0作为虚假raw_reward
                            is_real_list_.append(False)  # 标记为padding样本
                        
                        # 清理fake tensors
                        del fake_p_mask, fake_masked_indices, fake_labels, fake_valid_indices, fake_step_ids
                        
                        if accelerator.is_main_process and b < 2:
                            print(f"  Sample {b}: generated {round_count} training rounds, padded {padding_count} to {block_size}")
                    elif accelerator.is_main_process and b < 2:
                        print(f"  Sample {b}: generated {round_count} training rounds")
                    
                    if round_count == 0 and accelerator.is_main_process:
                        print(f"警告: Sample {b} 没有生成任何训练数据")
            else:
                raise ValueError(f"Unknown training.method: {config.training.method}")

            # 检查是否有有效的训练数据
            if len(input_ids_list) == 0:
                if accelerator.is_main_process:
                    print(f"跳过当前 batch：没有生成任何有效的训练数据")
                return None, None, None, None, None, None, [], [], [], 0

            input_ids_batch = torch.stack(input_ids_list, dim=0)  # (N, L)
            p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)  # (N, L)
            masked_indices = torch.stack(masked_indices_list, dim=0).to(torch.bool)  # (N, L)
            labels_batch = torch.stack(labels_list, dim=0)  # (N, L)
            valid_indices_batch = torch.stack(valid_indices_list, dim=0)  # (N, L)
            step_ids_batch = torch.stack(step_ids_list, dim=0)  # (N, L) - 添加 step_ids
            
            # 清理中间变量并释放显存
            del input_ids_list, pmask_list, masked_indices_list, labels_list, valid_indices_list, step_ids_list, step_map
            
            N = input_ids_batch.shape[0]
            if accelerator.is_main_process:
                print(f"  Generated {N} training samples from {B} original samples ({num_real_samples} real, {N - num_real_samples} padded)")
                print(f"===================================\n")
            
            # 返回前强制清理显存，避免累积到后面wakeup阶段
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return input_ids_batch, p_mask, masked_indices, valid_indices_batch, labels_batch, step_ids_batch, reward_list_, raw_reward_list_, is_real_list_, num_real_samples


        ##################################
        #        准备训练输入 - Response级别平铺 #
        ##################################
        # 先收集本地数据
        local_prompt_ids, local_token_ids, local_rewards, local_raw_rewards, local_step_maps = [], [], [], [], []
        
        for x in dataset_load:
            local_prompt_ids.extend(x["prompt_ids"])
            local_token_ids.extend(x["token_ids"])
            local_rewards.extend(x["rewards"])
            local_raw_rewards.extend(x["raw_rewards"])
            
            if "step_map" in x.keys() and x['step_map'] is not None:
                local_step_maps.extend(x["step_map"])
            else:
                for response_ids in x["token_ids"]:
                    local_step_maps.append(list(range(1, len(response_ids) + 1)))
        
        # Gather所有rank的response数据
        from accelerate.utils import gather_object
        all_responses = gather_object([{
            'prompt_ids': local_prompt_ids,
            'token_ids': local_token_ids,
            'rewards': local_rewards,
            'raw_rewards': local_raw_rewards,
            'step_maps': local_step_maps,
        }])
        
        # 在rank0上平铺重分配
        if accelerator.is_main_process:
            # 合并所有response
            all_prompt_ids = sum([d['prompt_ids'] for d in all_responses], [])
            all_token_ids = sum([d['token_ids'] for d in all_responses], [])
            all_rewards = sum([d['rewards'] for d in all_responses], [])
            all_raw_rewards = sum([d['raw_rewards'] for d in all_responses], [])
            all_step_maps = sum([d['step_maps'] for d in all_responses], [])
            
            total_responses = len(all_prompt_ids)
            responses_per_rank = (total_responses + accelerator.num_processes - 1) // accelerator.num_processes
            
            logger.info(f"Total responses: {total_responses}, distributing {responses_per_rank} per rank")
            
            # 按response平铺分配
            rank_data_list = []
            for rank_idx in range(accelerator.num_processes):
                start_idx = rank_idx * responses_per_rank
                end_idx = min((rank_idx + 1) * responses_per_rank, total_responses)
                
                rank_data_list.append({
                    'prompt_ids': all_prompt_ids[start_idx:end_idx],
                    'token_ids': all_token_ids[start_idx:end_idx],
                    'rewards': all_rewards[start_idx:end_idx],
                    'raw_rewards': all_raw_rewards[start_idx:end_idx],
                    'step_maps': all_step_maps[start_idx:end_idx],
                })
        else:
            rank_data_list = None
        
        # Broadcast分配结果
        from accelerate.utils import broadcast_object_list
        rank_data_list = broadcast_object_list([rank_data_list])[0]
        
        # 每个rank获取自己的数据
        rank_data = rank_data_list[accelerator.process_index]
        prompt_ids_list = rank_data['prompt_ids']
        token_ids_list = rank_data['token_ids']
        reward_list = rank_data['rewards']
        raw_reward_list = rank_data['raw_rewards']
        rollout_step_maps = rank_data['step_maps']
        
        logger.info(f"[Rank {accelerator.process_index}] After redistribution: {len(prompt_ids_list)} responses")
        accelerator.wait_for_everyone()

        # 调用新的 uni_prompting，传入step_map进行统一处理（包含截断和padding）
        input_ids_lm, labels_lm, valid_indices, step_map_tensor = uni_prompting(
            (prompt_ids_list, token_ids_list), 
            step_map_list=rollout_step_maps
        )
        
        B, L = input_ids_lm.shape
        print(f"[Rank {accelerator.process_index}] Batch size={B}, Seq length={L}")

        ##################################
        #     forward_process（新逻辑）   #
        ##################################
        def forward_process(input_ids, p_mask, masked_indices, valid_indices, labels, adv, logp_old_tok, is_real, step_ids, logp_ref_tok=None):
            import time
            time_dict = {}
            time_start_total = time.time()
            
            # 基本检查
            assert input_ids is not None and p_mask is not None and masked_indices is not None
            assert labels is not None and adv is not None and is_real is not None
            
            device = accelerator.device
            B, L = input_ids.shape
            
            # 转到GPU
            time_data_to_gpu_start = time.time()
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            p_mask = p_mask.to(device)
            masked_indices = masked_indices.to(device)
            valid_indices = valid_indices.to(device)
            step_ids = step_ids.to(device)
            if logp_old_tok is not None:
                logp_old_tok = logp_old_tok.to(device)
            # Check if logp_ref_tok is empty tensor (when KL=0) and convert to None
            if logp_ref_tok is not None:
                if logp_ref_tok.numel() == 0:
                    logp_ref_tok = None
                else:
                    logp_ref_tok = logp_ref_tok.to(device)
            time_dict['data_to_gpu'] = time.time() - time_data_to_gpu_start
            
            # 构建position_ids
            # valid_indices: 1=prompt, 2=response, 3=mask_padding, 0=padding
            # mask_padding(3)应该参与position计算（模拟最后一个block），但只有padding(0)的position设为0
            position_ids = torch.arange(L, device=device).long().unsqueeze(0).expand(B, -1)
            position_ids = torch.where(valid_indices == 0, torch.zeros_like(position_ids), position_ids)
            
            # 直接调用模型计算RL loss（使用预先计算好的ref logprob）
            time_model_start = time.time()
            outputs = model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                labels=labels,
                masked_indices=masked_indices,
                compute_rl_loss=True,
                p_mask=p_mask,
                adv=adv,
                logp_old_tok=logp_old_tok,
                logp_ref_tok=logp_ref_tok,  # 使用预先计算好的ref logprob
                dynamic_threshold=config.rollout.dynamic_threshold,
                step_ids=step_ids,
                is_real=is_real,
                ppo_eps=config.training.eps,
                kl_beta=config.training.get("beta", 0.0),
                use_kl_estimator_k3=config.training.get("use_kl_estimator_k3", True),
                return_entropy=True,
            )
            time_dict['model_forward'] = time.time() - time_model_start
            
            policy_loss = outputs.loss
            entropy_mean = outputs.entropy if hasattr(outputs, 'entropy') else torch.tensor(0.0, device=device)
            kl_loss = outputs.kl_loss if hasattr(outputs, 'kl_loss') else torch.tensor(0.0, device=device)
            
            # 清理outputs以释放显存
            del outputs
            
            time_dict['forward_total'] = time.time() - time_start_total

            return policy_loss, entropy_mean, kl_loss, time_dict



        ##################################
        #         训练循环（动态扩展）    #
        ##################################
        from tqdm.auto import tqdm

        logger.info("***** Running training (dynamic per-batch expansion) *****")
        logger.info(f"  Num response = {len(dataset_load)}")
        logger.info(f"  Num original samples = {input_ids_lm.shape[0]}")
        num_train_steps = config.training.num_train_steps
        num_iterations = config.training.get("num_iterations", 8)
        logger.info(f"  Number of iterations over same data: {num_iterations}")

        total_samples = input_ids_lm.shape[0]
        batch_size_train = config.training.batch_size_lm

        ##################################
        #   准备训练数据                  #
        ##################################
        logger.info("***** Preparing training data *****")
        
        logger.info(f"[Rank {accelerator.process_index}] Total response samples: {total_samples}")
        logger.info(f"[Rank {accelerator.process_index}] Gradient accumulation steps: {gradient_accumulation_steps}")

        # 加载训练模型
        logger.info("=" * 80)
        logger.info("Loading training model")
        logger.info("=" * 80)
            
        # 加载模型
        load_start_time = time.time()
        model_already_loaded = False
        
        # Step 1: 加载参考模型（如果启用KL）
        if config.training.get("beta", 0) > 0:
            logger.info("=" * 80)
            logger.info("Step 1: Loading reference model for KL divergence")
            logger.info("=" * 80)
            
            if ref_model is None:
                # 第一个step，加载参考模型
                if step == config.experiment.current_step and config.experiment.current_step == 1:
                    logger.info(f"[Rank {accelerator.process_index}] Loading model from {pretrained_model} for both training and reference")
                    
                    if dist.get_rank() == 0:
                        model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
                    dist.barrier()
                    if dist.get_rank() != 0:
                        model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
                    
                    if hasattr(model, "config"):
                        model.config.fuse_cross_entropy = False
                    
                    # 克隆模型作为参考模型
                    import copy
                    logger.info(f"[Rank {accelerator.process_index}] Cloning model as reference model")
                    ref_model = copy.deepcopy(model)
                    ref_model.eval()
                    # 冻结参考模型参数
                    for param in ref_model.parameters():
                        param.requires_grad = False
                    
                    # 训练模型设置
                    if config.training.gradient_checkpointing_enable:
                        model.gradient_checkpointing_enable()
                        if hasattr(model, "config"):
                            model.config.use_cache = False
                    
                    model_already_loaded = True
                else:
                    # 续训或后续step，正常加载参考模型
                    ref_model_path = config.model.pretrained_model
                    logger.info(f"[Rank {accelerator.process_index}] Loading reference model from {ref_model_path}")
                    
                    if dist.get_rank() == 0:
                        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, trust_remote_code=True, torch_dtype="auto")
                    dist.barrier()
                    if dist.get_rank() != 0:
                        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, trust_remote_code=True, torch_dtype="auto")
                    
                    if hasattr(ref_model, "config"):
                        ref_model.config.fuse_cross_entropy = False
                    ref_model.eval()
                    # 冻结参考模型参数
                    for param in ref_model.parameters():
                        param.requires_grad = False
            else:
                logger.info(f"[Rank {accelerator.process_index}] Reference model already loaded, will move to GPU when needed")
            
            # 不在这里移动到GPU，而是在每次迭代需要计算时才移动
            logger.info(f"[Rank {accelerator.process_index}] Reference model loaded successfully (on CPU)")
        else:
            logger.info("KL penalty disabled (beta=0), skipping reference model loading")
        
        # Step 2: 加载训练模型（如果还没加载）
        logger.info("=" * 80)
        logger.info("Step 2: Loading training model")
        logger.info("=" * 80)
        
        if step == config.experiment.current_step and not model_already_loaded:
            logger.info(f"[Rank {accelerator.process_index}] Loading training model from {pretrained_model}")
            if dist.get_rank() == 0:
                model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
            dist.barrier()
            if dist.get_rank() != 0:
                model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

            # calculate loss ourselves, needs logits，so aviod fuse CE
            if hasattr(model, "config"):
                model.config.fuse_cross_entropy = False

            # Setup gradient checkpointing
            if config.training.gradient_checkpointing_enable:
                logger.info("Enabling gradient checkpointing")
                model.gradient_checkpointing_enable()
                if hasattr(model, "config"):
                    model.config.use_cache = False
            else:
                model = model.to(accelerator.device)
        else:
            # 后续step，从CPU移到GPU
            model = model.to(accelerator.device)
        
        load_time = time.time() - load_start_time
        time_stats['load'] = load_time
        logger.info(f"[Time Stats] Load operation completed in {load_time:.2f}s")


        ##################################
        #   Optimizer and LR scheduler   #
        #################################
        if step == config.experiment.current_step:
            logger.info("Initializing optimizer and LR scheduler")
            optimizer_config = config.optimizer.params

            # no decay on bias and layernorm and embedding
            no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               p.requires_grad and not any(nd in n for nd in no_decay)],
                    "weight_decay": optimizer_config.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               p.requires_grad and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_type = config.optimizer.name
            if optimizer_type == "adamw":
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=optimizer_config.learning_rate,
                    betas=(optimizer_config.beta1, optimizer_config.beta2),
                    weight_decay=optimizer_config.weight_decay,
                    eps=optimizer_config.epsilon,
                )
            else:
                raise ValueError(f"Optimizer {optimizer_type} not supported")

        ##################################
        #   LR Scheduler & Accelerator   #
        ##################################
        num_batches_train = input_ids_lm.shape[0]
        max_train_iters = num_iterations * num_train_steps
        if step == config.experiment.current_step:
            logger.info("Initializing LR scheduler and preparing with accelerator")
            lr_scheduler = get_scheduler(
                config.lr_scheduler.scheduler,
                optimizer=optimizer,
                num_training_steps=max_train_iters,
                num_warmup_steps=config.lr_scheduler.params.warmup_steps,
                min_lr_scale=config.lr_scheduler.params.min_lr_scale
            )

            dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1, dtype=torch.long))
            dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1)
            # 注意：lr_scheduler 不要被 accelerator.prepare() 包装，否则会导致 current_step 累加 world_size 倍
            model, optimizer, dummy_loader = accelerator.prepare(model, optimizer, dummy_loader)
        else:
            dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1, dtype=torch.long))
            dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1)
            dummy_loader = accelerator.prepare(dummy_loader)
        
        # 确保模型处于训练模式
        model.train()

        logger.info(f"[Rank {accelerator.process_index}] Model loaded and ready for training.")

        ##################################
        #         训练循环（预收集+批处理）#
        ##################################
            
        ##################################
        # 数据预处理（只做一次，iteration共享）#
        ##################################
        logger.info(f"[Rank {accelerator.process_index}] Collecting expanded training data...")
        time_collect_start = time.time()
        
        # 一次性处理所有样本（只做一次！）
        expanded_data = collect_training_data(
            input_ids_lm,      # (total_samples, L)
            labels_lm,         # (total_samples, L)
            valid_indices,     # (total_samples, L)
            step_map_tensor,   # (total_samples, L) tensor
            reward_list,       # List 长度为 total_samples
            raw_reward_list    # List 长度为 total_samples
        )
        
        # 检查是否有有效数据
        if expanded_data[0] is None:
            logger.warning(f"[Rank {accelerator.process_index}] No valid training data generated, skipping this step")
            continue
        
        expanded_input_ids, expanded_p_mask, expanded_masked_indices, expanded_valid_indices, expanded_labels, expanded_step_ids, all_rewards, all_raw_rewards, all_is_real, num_real_samples = expanded_data
        
        total_expanded_samples = expanded_input_ids.shape[0]
        num_padded_samples = total_expanded_samples - num_real_samples
        time_collect = time.time() - time_collect_start
        
        logger.info(f"[Rank {accelerator.process_index}] Collected {total_expanded_samples} expanded samples in {time_collect:.2f}s "
                   f"(real: {num_real_samples}, padded: {num_padded_samples})")
        
        # total_samples = 当前rank分到的response数（rollout时已按response平铺分配）
        num_response_per_task = config.rollout.num_response_per_task
        num_local_responses = total_samples
        num_local_tasks = num_local_responses // num_response_per_task
        
        logger.info(f"[Rank {accelerator.process_index}] Local: {num_local_responses} responses, {num_local_tasks} tasks")
        
        ##################################
        # 通用函数：计算模型的logprobs
        ##################################
        def compute_model_logprobs(model, expanded_input_ids, expanded_masked_indices, 
                                   expanded_valid_indices, expanded_labels, expanded_step_ids, 
                                   batch_size, model_name="model"):
            """通用函数：计算模型在所有expanded数据上的log probabilities"""
            logger.info(f"[Rank {accelerator.process_index}] Computing {model_name} logprobs...")
            
            # 创建临时DataLoader
            temp_dataset = torch.utils.data.TensorDataset(
                expanded_input_ids, expanded_masked_indices, 
                expanded_valid_indices, expanded_labels, expanded_step_ids
            )
            temp_loader = torch.utils.data.DataLoader(
                temp_dataset, batch_size=batch_size, shuffle=False
            )
            
            # 设置为train模式（走训练分支支持masked_indices），但用no_grad不计算梯度
            model.train()
            
            logprobs_list = []
            with torch.no_grad():
                for batch_data in temp_loader:
                    b_input_ids, b_masked_indices, b_valid_indices, b_labels, b_step_ids = batch_data
                    B, L = b_input_ids.shape
                    device = accelerator.device
                    
                    # 转到GPU
                    b_input_ids = b_input_ids.to(device)
                    b_masked_indices = b_masked_indices.to(device)
                    b_valid_indices = b_valid_indices.to(device)
                    b_labels = b_labels.to(device)
                    b_step_ids = b_step_ids.to(device)
                    
                    # 构建position_ids
                    position_ids = torch.arange(L, device=device).long().unsqueeze(0).expand(B, -1)
                    position_ids = torch.where(b_valid_indices == 0, torch.zeros_like(position_ids), position_ids)
                    
                    # 模型forward
                    outputs = model(
                        input_ids=b_input_ids,
                        attention_mask=None,  # 训练分支会自己构建flex attention mask
                        position_ids=position_ids,
                        labels=b_labels,
                        masked_indices=b_masked_indices,
                        return_logits=True,
                        step_ids=b_step_ids,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    
                    # 计算log概率
                    logits = outputs.logits  # (M, V)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # 创建完整的logp张量 (B, L)
                    logp_batch = torch.zeros(B, L, device=device, dtype=logits.dtype)
                    labels_masked = b_labels[b_masked_indices]
                    logp_masked = log_probs.gather(dim=-1, index=labels_masked.unsqueeze(-1)).squeeze(-1)
                    logp_batch[b_masked_indices] = logp_masked
                    
                    # 保存到CPU
                    logprobs_list.append(logp_batch.cpu())
                    
                    # 清理
                    del outputs, logits, log_probs, logp_masked, labels_masked, logp_batch
                    del b_input_ids, b_masked_indices, b_valid_indices, b_labels, b_step_ids, position_ids
                    torch.cuda.empty_cache()
            
            # 拼接所有batch
            all_logprobs = torch.cat(logprobs_list, dim=0)
            del logprobs_list, temp_dataset, temp_loader
            torch.cuda.empty_cache()
            
            logger.info(f"[Rank {accelerator.process_index}] {model_name} logprobs computed")
            return all_logprobs
        
        ##################################
        # 计算参考模型logprob（每个epoch只做一次！）#
        ##################################
        expanded_ref_logprobs = None
        if config.training.get("beta", 0) > 0 and ref_model is not None:
            time_ref_start = time.time()
            
            # 将ref模型移到GPU
            if next(ref_model.parameters()).device != accelerator.device:
                logger.info(f"[Rank {accelerator.process_index}] Moving reference model to GPU")
                ref_model = ref_model.to(accelerator.device)
            
            # 计算ref logprobs
            expanded_ref_logprobs = compute_model_logprobs(
                ref_model, expanded_input_ids, expanded_masked_indices,
                expanded_valid_indices, expanded_labels, expanded_step_ids,
                batch_size_train, "reference"
            )
            
            time_ref = time.time() - time_ref_start
            logger.info(f"[Rank {accelerator.process_index}] Reference logprobs computed in {time_ref:.2f}s")
            
            # 清理并移回CPU
            ref_model.eval()
            ref_model = ref_model.cpu()
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            accelerator.wait_for_everyone()
        
        ##################################
        # 计算old policy logprobs（GRPO需要）#
        ##################################
        # 如果num_iterations=1且mini_batch_size等于num_task_per_step，不需要计算old_logprobs
        if num_iterations == 1 and mini_batch_size == num_task_per_step:
            expanded_old_logprobs = None
            logger.info(f"[Rank {accelerator.process_index}] Skipping old policy logprobs computation (num_iterations=1, mini_batch_size=num_task_per_step)")
        else:
            time_old_logp_start = time.time()
            
            # 计算old logprobs
            expanded_old_logprobs = compute_model_logprobs(
                model, expanded_input_ids, expanded_masked_indices,
                expanded_valid_indices, expanded_labels, expanded_step_ids,
                batch_size_train, "old policy"
            )
            
            time_old_logp = time.time() - time_old_logp_start
            logger.info(f"[Rank {accelerator.process_index}] Old policy logprobs computed in {time_old_logp:.2f}s")
        
        # 保持train模式继续训练
        model.train()
        torch.cuda.empty_cache()
        
        ##################################
        # 构建DataLoader（只做一次！）    #
        ##################################
        from torch.utils.data import DataLoader
        
        train_dataset = TrainDataset(
            expanded_input_ids,
            expanded_p_mask,
            expanded_masked_indices,
            expanded_valid_indices,
            expanded_labels,
            all_rewards,
            all_raw_rewards,
            all_is_real,
            expanded_step_ids,
            expanded_ref_logprobs,  # 传入预先计算好的ref logprob
            expanded_old_logprobs   # 传入预先计算好的old logprob (GRPO)
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=False,
        )
        
        num_forward_batches = len(train_dataloader)
        
        # 使用前面计算好的参数（无需重复定义）
        # mini_batch_size, num_response_per_task, num_processes 已在外层定义
        
        logger.info(f"[Rank {accelerator.process_index}] Mini-batch size: {mini_batch_size}")
        logger.info(f"[Rank {accelerator.process_index}] Training with {total_expanded_samples} expanded samples "
                   f"({num_forward_batches} base batches, gradient_accumulation_steps: {gradient_accumulation_steps}, "
                   f"real samples: {num_real_samples})")
        logger.info(f"[Rank {accelerator.process_index}] Effective batch size per step: {batch_size_train * mini_batch_size}")
        
        # Check if all ranks have the same num_forward_batches to avoid deadlock
        all_num_batches = accelerator.gather(torch.tensor([num_forward_batches], device=accelerator.device))
        if accelerator.is_main_process:
            unique_counts = all_num_batches.unique()
            if len(unique_counts) > 1:
                logger.error(f"Batch count mismatch across ranks: {all_num_batches.tolist()}")
                raise RuntimeError(f"All ranks must have the same num_forward_batches, got: {all_num_batches.tolist()}")
            logger.info(f"Verified: all ranks have {num_forward_batches} batches")
        accelerator.wait_for_everyone()
        
        # ========== 开始 Training 阶段计时 ==========
        training_start_time = time.time()
        total_training_tokens = 0  # 统计训练阶段处理的token数
        
        ##################################
        # Iteration循环（多次遍历同一数据）#
        ##################################
        # 使用前面计算好的gradient_accumulation_steps（已考虑expansion_factor）
        effective_accum_steps = gradient_accumulation_steps
        logger.info(f"[Rank {accelerator.process_index}] Using gradient_accumulation_steps: {effective_accum_steps}")
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration+1}/{num_iterations}")
            
            # Mini-batch统计变量
            mb_loss, mb_reward, mb_entropy, mb_kl = 0.0, 0.0, 0.0, 0.0
            mb_count = 0
            grad_norm = 0.0
            should_log = False  # 标记是否应该logging
            
            progress_bar = tqdm(enumerate(train_dataloader),
                                total=num_forward_batches,
                                desc=f"Step {step} Iter {iteration+1}/{num_iterations}",
                                disable=not accelerator.is_local_main_process)

            for bstep, batch in progress_bar:
                # 解包batch
                input_ids, p_mask, masked_indices, valid_indices, labels, rewards, raw_rewards, is_real, step_ids, ref_logp, old_logp = batch
                
                # Forward & Backward（每次都同步梯度）
                loss, entropy, kl, _ = forward_process(
                    input_ids, p_mask, masked_indices, valid_indices, labels,
                    rewards, old_logp, is_real, step_ids, ref_logp
                )
                
                # Backward，loss除以累积步数来平均
                accelerator.backward(loss / gradient_accumulation_steps)
                
                # 累积统计
                mb_loss += loss.item()
                mb_count += 1
                
                # 达到累积步数时才更新参数
                if (bstep + 1) % gradient_accumulation_steps == 0 or (bstep + 1) == num_forward_batches:
                    should_log = True
                    
                    # Clip梯度
                    max_grad_norm = config.training.get("max_grad_norm", float('inf'))
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # 获取本次更新使用的学习率（在step之前）
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Debug: 输出训练关键信息
                    if accelerator.is_main_process:
                        grad_str = f"{grad_norm:.4f}" if grad_norm is not None else "N/A"
                        print(f"[Optimizer Step {optimizer_step}] lr={current_lr}, grad_norm={grad_str}, loss={mb_loss/mb_count:.6f}")
                # 统计当前batch
                total_training_tokens += (labels != -100).sum().item()
                real_reward = sum([r for r, is_r in zip(raw_rewards, is_real) if is_r])
                real_count = sum(is_real)
                mb_reward += real_reward / max(real_count, 1)
                mb_entropy += entropy.item()
                mb_kl += kl.item()
                global_step += 1
                
                # 清理本batch的tensor
                del loss, entropy, kl
                del input_ids, p_mask, masked_indices, valid_indices, labels, rewards, raw_rewards, is_real, step_ids, ref_logp, old_logp
                
                # 每个batch后清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 每次真正执行optimizer.step()时（累积完成）进行logging
                if should_log:
                    # 聚合所有rank的统计值
                    mb_stats = torch.tensor([mb_loss, mb_reward, mb_entropy, mb_kl, float(mb_count)], device=accelerator.device)
                    all_stats = accelerator.gather(mb_stats).cpu().view(accelerator.num_processes, -1)
                    
                    # 计算全局平均
                    global_loss = all_stats[:, 0].sum() / all_stats[:, 4].sum()
                    global_reward = all_stats[:, 1].sum() / all_stats[:, 4].sum()
                    global_entropy = all_stats[:, 2].sum() / all_stats[:, 4].sum()
                    global_kl = all_stats[:, 3].sum() / all_stats[:, 4].sum()
                    
                    # Log（训练metrics是每次optimizer更新时记录，使用optimizer_step作为x轴）
                    grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm if grad_norm is not None else 0.0)
                    if accelerator.is_main_process:
                        logger.info(f"Optimizer Step {optimizer_step} (Training Step {step}) | Loss: {global_loss:.6f} | "
                                   f"Reward: {global_reward:.6f} | Entropy: {global_entropy:.6f} | "
                                   f"KL: {global_kl:.6f} | GradNorm: {grad_norm_value:.3f}")
                    
                    accelerator.log({
                        "train/loss": global_loss.item(),
                        "train/reward": global_reward.item(),
                        "train/entropy": global_entropy.item(),
                        "train/kl": global_kl.item(),
                        "train/grad_norm": grad_norm_value,
                        "train/lr": current_lr,
                        "train/optimizer_step": optimizer_step,  # 记录optimizer step
                    }, step=step)
                    
                    # 清理统计tensor
                    del mb_stats, all_stats, global_loss, global_reward, global_entropy, global_kl
                    
                    # 保存checkpoint
                    if config.training.get("save_steps") and optimizer_step % config.training.save_steps == 0:
                        save_checkpoint(model, tokenizer, config, accelerator, f"step-{step}-iter-{optimizer_step}")
                    
                    # 更新optimizer_step计数
                    optimizer_step += 1
                    
                    # 重置mini-batch统计
                    mb_loss, mb_reward, mb_entropy, mb_kl = 0.0, 0.0, 0.0, 0.0
                    mb_count = 0
                    should_log = False
                    
                    # 每次更新后清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        
        # 清理训练数据（iteration循环结束后）
        del train_dataloader, train_dataset
        del expanded_input_ids, expanded_p_mask, expanded_masked_indices
        del expanded_valid_indices, expanded_labels, expanded_step_ids
        del all_rewards, all_raw_rewards, all_is_real
        if expanded_ref_logprobs is not None:
            del expanded_ref_logprobs
        
        # 强制清理显存，避免累积到 wakeup 阶段
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        
        # ========== 结束 Training 阶段计时 ==========
        training_time = time.time() - training_start_time
        time_stats['training'] = training_time
        
        # 计算训练吞吐量（TGS: tokens per GPU per second）
        from accelerate.utils import gather_object
        all_training_data = gather_object([{
            'tokens': total_training_tokens,
            'time': training_time
        }])
        
        global_training_tokens = sum(d['tokens'] for d in all_training_data)
        num_gpus = accelerator.num_processes
        # TGS = 总tokens / (最慢GPU时间 * GPU数)
        max_training_time = max((d['time'] for d in all_training_data), default=0.0)
        training_tgs = global_training_tokens / (max_training_time * num_gpus) if max_training_time > 0 else 0.0
        
        time_stats['training_tokens'] = global_training_tokens
        time_stats['training_throughput'] = training_tgs
        
        logger.info(f"[Time Stats] Training phase completed in {training_time:.2f}s")
        if accelerator.is_main_process:
            logger.info(f"[Throughput] Training TGS: {global_training_tokens:,} tokens / ({max_training_time:.2f}s * {num_gpus} GPUs) = {training_tgs:.2f} tokens/gpu/s")

        logger.info(f"Step {step} training completed")
        logger.info(f"Total optimizer updates: {optimizer_step}")

        logger.info(f"[Rank {accelerator.process_index}] Waking up server.")

        # server_sleep(base_url, level=1)

        # 监控显存状态
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"[GPU {i}] Before update_weights: "
                          f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, "
                          f"Max: {max_allocated:.2f}GB, Total: {total:.2f}GB")
        
        # 强制清理显存
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # 再次监控
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"[GPU {i}] After cleanup: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # wakeup weights before update
        wakeup_weights_start_time = time.time()
        server_wakeup(base_url, tags=['weights'])
        wakeup_weights_time = time.time() - wakeup_weights_start_time
        time_stats['wakeup_weights'] = wakeup_weights_time
        print(f"[Time Stats] Wakeup(weights): {wakeup_weights_time:.2f}s")

        # update weights 
        update_start_time = time.time()
        model_unwrapped = accelerator.unwrap_model(model)
        server_update_weights(base_url, model_unwrapped)
        update_time = time.time() - update_start_time
        time_stats['update'] = update_time
        print(f"[Time Stats] Update weights: {update_time:.2f}s")
        
        # 清理 unwrapped 引用，释放显存
        del model_unwrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        if step % config.experiment.save_every == 0:
            # ========== 开始 Save 阶段计时 ==========
            save_start_time = time.time()
            logger.info(f"Saving step checkpoint: step-{step}")
            save_checkpoint(model, tokenizer, config, accelerator, f"step-{step}")
            # ========== 结束 Save 阶段计时 ==========
            save_time = time.time() - save_start_time
            time_stats['save'] = save_time
            print(f"[Time Stats] Save: {save_time:.2f}s")
        else:
            time_stats['save'] = 0.0

        # 在 wakeup 之前，将训练模型和optimizer移到 CPU 释放显存
        logger.info(f"[Rank {accelerator.process_index}] Moving model to CPU to free GPU memory for inference server")
        model = model.cpu()
        
        # 将optimizer状态移到CPU (已取消此操作)
        # if optimizer is not None:
        #     opt_to_cpu_start = time.time()
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.cpu()
        #     opt_to_cpu_time = time.time() - opt_to_cpu_start
        #     logger.info(f"[Rank {accelerator.process_index}] Moved optimizer state to CPU in {opt_to_cpu_time:.2f}s")
        
        # 同时将参考模型移到CPU（如果存在）
        if ref_model is not None:
            logger.info(f"[Rank {accelerator.process_index}] Moving reference model to CPU")
            ref_model = ref_model.cpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wakeup_kv_start_time = time.time()
        server_wakeup(base_url, tags=['kv_cache'])
        wakeup_kv_time = time.time() - wakeup_kv_start_time
        time_stats['wakeup_kv'] = wakeup_kv_time
        print(f"[Time Stats] Wakeup(kv_cache): {wakeup_kv_time:.2f}s")
        
        accelerator.wait_for_everyone()
        
        # Evaluation（推理服务器运行时，训练模型保持在 CPU）
        eval_every = config.experiment.get("eval_every", 1)
        if step % eval_every == 0:
            # ========== 开始 Eval 阶段计时 ==========
            eval_start_time = time.time()
            logger.info(f"Starting evaluation on {config.dataset.eval_dataset}...")
            eval_model_path = config.experiment.prefix_dir + project_name + f"/ckpt/step-{step}"
            
            eval_data = rollout_sampling_eval(
                client,
                config.dataset.eval_dataset, step, config,
                eval_model_path, model_name, tokenizer, accelerator,
                normalize=False, filter=False, mode="eval"
            )
            
            # No need to sleep after evaluation as next step will wakeup anyway

            total_correct = 0
            total_samples_eval = 0
            local_eval_tokens = 0
            
            for item in eval_data:
                total_correct += sum(item["correctness"])
                total_samples_eval += len(item["correctness"])
                token_lengths = item.get("token_length", [])
                local_eval_tokens += sum(token_lengths)

            accuracy = total_correct / total_samples_eval if total_samples_eval > 0 else 0.0
            
            # ========== 结束 Eval 阶段计时 ==========
            eval_time = time.time() - eval_start_time
            time_stats['eval'] = eval_time
            
            # Gather所有rank的统计
            from accelerate.utils import gather_object
            all_eval_data = gather_object([{
                'tokens': local_eval_tokens,
                'time': eval_time,
                'correct': total_correct,
                'samples': total_samples_eval
            }])
            
            global_eval_tokens = sum(d['tokens'] for d in all_eval_data)
            max_eval_time = max(d['time'] for d in all_eval_data)
            cluster_eval_tps = global_eval_tokens / max_eval_time if max_eval_time > 0 else 0.0
            global_eval_correct = sum(d['correct'] for d in all_eval_data)
            global_eval_samples = sum(d['samples'] for d in all_eval_data)
            global_accuracy = global_eval_correct / global_eval_samples if global_eval_samples > 0 else 0.0
            
            time_stats['eval_tokens'] = global_eval_tokens
            time_stats['eval_throughput'] = cluster_eval_tps
            
            logger.info(f"[Time Stats] Eval phase completed in {eval_time:.2f}s")
            if accelerator.is_main_process:
                logger.info(f"[Throughput] Cluster TPS: {global_eval_tokens:,} tokens / {max_eval_time:.2f}s = {cluster_eval_tps:.2f} tokens/s")
                logger.info(f"Evaluation accuracy: {global_accuracy*100:.2f}% ({global_eval_correct}/{global_eval_samples})")

            # Evaluation metrics是每个step执行一次，使用step作为x轴
            accelerator.log({
                "eval/accuracy": global_accuracy,
                "eval/total_correct": global_eval_correct,
                "eval/total_samples": global_eval_samples,
                "eval/step": step,
            }, step=step)
        else:
            time_stats['eval'] = 0.0
            time_stats['eval_tokens'] = 0
            time_stats['eval_throughput'] = 0.0

        # ========== Step总时间统计 ==========
        step_total_time = time.time() - step_start_time
        time_stats['step_total'] = step_total_time
        
        # 输出时间统计摘要
        logger.info(f"\n{'='*80}")
        logger.info(f"Time Statistics for Step {step}")
        logger.info(f"{'='*80}")
        logger.info(f"  Rollout:  {time_stats['rollout']:.2f}s ({time_stats['rollout']/step_total_time*100:.1f}%)")
        logger.info(f"  Training: {time_stats['training']:.2f}s ({time_stats['training']/step_total_time*100:.1f}%)")
        logger.info(f"  Save:     {time_stats['save']:.2f}s ({time_stats['save']/step_total_time*100:.1f}%)")
        logger.info(f"  Eval:     {time_stats['eval']:.2f}s ({time_stats['eval']/step_total_time*100:.1f}%)")
        logger.info(f"  Total:    {step_total_time:.2f}s")
        logger.info(f"{'='*80}")
        logger.info(f"Server Operations Time Statistics")
        logger.info(f"{'='*80}")
        logger.info(f"  Load:     {time_stats.get('load', 0):.2f}s ({time_stats.get('load', 0)/step_total_time*100:.1f}%)")
        logger.info(f"  Sleep:    {time_stats.get('sleep', 0):.2f}s ({time_stats.get('sleep', 0)/step_total_time*100:.1f}%)")
        logger.info(f"  Update:   {time_stats.get('update', 0):.2f}s ({time_stats.get('update', 0)/step_total_time*100:.1f}%)")
        logger.info(f"  Wakeup:   {time_stats.get('wakeup', 0):.2f}s ({time_stats.get('wakeup', 0)/step_total_time*100:.1f}%)")
        logger.info(f"{'='*80}")
        logger.info(f"Throughput Statistics for Step {step}")
        logger.info(f"{'='*80}")
        logger.info(f"  Sampling:  {time_stats.get('rollout_tokens', 0):,} tokens, {time_stats.get('sampling_throughput', 0):.2f} TPS (cluster)")
        logger.info(f"  Training:  {time_stats.get('training_tokens', 0):,} tokens, {time_stats.get('training_throughput', 0):.2f} TGS (per GPU)")
        if time_stats.get('eval_tokens', 0) > 0:
            logger.info(f"  Eval:      {time_stats.get('eval_tokens', 0):,} tokens, {time_stats.get('eval_throughput', 0):.2f} TPS (cluster)")
        logger.info(f"{'='*80}\n")
        
        # 记录到 wandb（时间统计是每个step统计一次，使用step作为x轴）
        if accelerator.is_main_process:
            accelerator.log({
                "time/rollout": time_stats['rollout'],
                "time/training": time_stats['training'],
                "time/save": time_stats['save'],
                "time/eval": time_stats['eval'],
                "time/step_total": step_total_time,
                "time/rollout_percent": time_stats['rollout']/step_total_time*100,
                "time/training_percent": time_stats['training']/step_total_time*100,
                "time/save_percent": time_stats['save']/step_total_time*100,
                "time/eval_percent": time_stats['eval']/step_total_time*100,
                "time/step": step,
                "time/optimizer_step": optimizer_step,  # 同时记录optimizer_step以便对照
            }, step=step)

        logger.info(f"Step {step} completed successfully\n")

        accelerator.wait_for_everyone()


    # End training after all steps are complete
    logger.info(f"{'='*80}")
    logger.info(f"All training completed! Total steps: {config.training.num_train_steps}")
    logger.info(f"{'='*80}")
    
    accelerator.end_training()



def main():
    train()

def save_checkpoint(model, tokenizer, config, accelerator, name):
    """保存checkpoint"""
    from pathlib import Path
    import time, json, shutil, os, glob

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        save_dir = save_base / name

        # 直接从 model_to_save 抽取权重，避免 get_state_dict 丢参数
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(save_dir)

        # 保存当前 config，而不是覆盖旧的
        model_to_save.config.save_pretrained(save_dir)

        # 复制自定义源码文件
        model_path = config.model.pretrained_model
        for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py", "fused_linear_*.py"):
            for fn in glob.glob(os.path.join(model_path, pattern)):
                dst = os.path.join(save_dir, os.path.basename(fn))
                if not os.path.exists(dst):
                    shutil.copy2(fn, dst)

        # 记录元数据
        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": name
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved checkpoint to {save_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
