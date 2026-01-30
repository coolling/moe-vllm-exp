# eval_vllm_sequential.py
"""
使用 vLLM 逐条（非批量）方式，在 8 个 zero-shot 任务上评估模型。
适用于复现论文结果（如 Mixtral 剪枝实验）。
"""

import os
from typing import List, Tuple, Any
from vllm import LLM, SamplingParams
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import evaluator, tasks
from lm_eval.api.instance import Instance
import psutil
BIND_CPUS = [40, 41,42,43,44,45,46,47,48,49,50,51]  
# 方式1：用psutil（跨平台，Windows/Linux/Mac都支持，推荐）
if BIND_CPUS:
    p = psutil.Process(os.getpid())
    p.cpu_affinity(BIND_CPUS)  # 绑定当前进程到指定CPU核心
    print(f"已绑定当前评测进程到CPU核心：{BIND_CPUS}")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "10"
# os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "40-51"
# ==============================
# 1. 自定义 LM 类：强制逐条处理
# ==============================
@register_model("vllm_sequential")
class VLLMSequential(LM):
    def __init__(
        self,
        pretrained: str,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        max_model_len: int = None,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        super().__init__()
        print(f"[VLLM] Loading model: {pretrained}")
        self.llm = LLM(
            model=pretrained,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=1,  # ⭐ 关键：禁用批处理，只允许1个序列并发
            enforce_eager=True,
            **kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        for instance in requests:
            context, continuation = instance.args  # ← 关键修改！
            print(context, continuation)
            prompt = context + continuation
            context_ids = self.tokenizer.encode(context, add_special_tokens=False)
            full_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            cont_start_idx = len(context_ids)

            if cont_start_idx >= len(full_ids):
                res.append((0.0, True))
                continue

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                prompt_logprobs=0,  # 获取 prompt 中每个 token 的 logprob
                detokenize=True,
                skip_special_tokens=False,
                spaces_between_special_tokens=False
            )

            # ⭐ 单条推理
            output = self.llm.generate([prompt], sampling_params, use_tqdm=False)[0]
            prompt_logprobs = output.prompt_logprobs  # list[Optional[dict]]

            if prompt_logprobs is None:
                raise RuntimeError("prompt_logprobs is None. Check vLLM version.")

            logprob_sum = 0.0
            is_greedy = True

            # prompt_logprobs[i] 对应 full_ids[i]（i 从 0 开始，但 logprobs[0] 是 pre-first，通常为 None）
            for i in range(cont_start_idx, len(full_ids)):
                token_id = full_ids[i]
                # 注意：logprobs 索引 = i（因为 logprobs 长度 = len(full_ids)）
                logprob_dict = prompt_logprobs[i]
                if logprob_dict is None:
                    continue  # 跳过第一个位置（通常是 BOS）

                if token_id in logprob_dict:
                    logprob_obj = logprob_dict[token_id]
                    logprob_sum += logprob_obj.logprob
                    if logprob_obj.rank != 1:
                        is_greedy = False
                else:
                    # Fallback to top logprob
                    top_token = max(logprob_dict, key=lambda k: logprob_dict[k].logprob)
                    logprob_sum += logprob_dict[top_token].logprob
                    is_greedy = False

            res.append((logprob_sum, is_greedy))
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        for instance in requests:
            prompt, gen_kwargs = instance.args  # ✅ 提取 (str, dict)
            print(prompt, gen_kwargs)
            # 你的生成逻辑...
            sampling_params = SamplingParams(
                temperature=gen_kwargs.get("temperature", 0.0),
                top_p=gen_kwargs.get("top_p", 1.0),
                max_tokens=gen_kwargs.get("max_gen_toks", 256),
                stop=gen_kwargs.get("until", None),
                detokenize=True,
            )
            output = self.llm.generate([prompt], sampling_params, use_tqdm=False)[0]
            generated_text = output.outputs[0].text
            res.append(generated_text)
        
        return res

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError("Not needed for standard zero-shot tasks.")


# ==============================
# 2. 主评估函数
# ==============================
def main():
    # 🔧 配置区 —— 根据你的环境修改
    MODEL_NAME = "/sharenvme/usershome/cyl/test/model/mistralai/Mixtral-8x7B-Instruct-v0.1"  # 或你的本地模型路径
    DTYPE = "bfloat16"          # 可选: "float16", "bfloat16", "auto"
    MAX_MODEL_LEN = 4096        # 最大上下文长度

    # 论文中的 8 个 zero-shot 任务
    TASKS = [
    "gsm8k",      # 主力任务（必做）
    "mbpp",       # 进阶验证（可选）
    "ifeval", # 代码场景验证（可选）
    "xsum", 
    "cnn_dailymail"
]
    print(f"\n🚀 Starting zero-shot evaluation on {len(TASKS)} tasks...")
    print(f"Model: {MODEL_NAME}")
    print(f"Tasks: {', '.join(TASKS)}\n")

    # 初始化任务注册表
    # tasks.initialize_tasks()

    # 运行评估（使用我们注册的 vllm_sequential 模型）
    results = evaluator.simple_evaluate(
        model="vllm_sequential",
        model_args=f"pretrained={MODEL_NAME},dtype={DTYPE},max_model_len={MAX_MODEL_LEN}",
        tasks=TASKS,
        num_fewshot=0,       # ⭐ Zero-shot
        # batch_size=1,        # 虽然设为1，但我们内部已逐条处理
        limit=1,          # 评估完整数据集；测试时可设为 10
        log_samples=False,
        confirm_run_unsafe_code=True,
        # use_cache=False      # 不缓存结果（避免污染）
    )

    # ==============================
    # 3. 打印结果 & 计算平均分
    # ==============================
    print("\n" + "="*60)
    print("📊 ZERO-SHOT EVALUATION RESULTS")
    print("="*60)

    task_scores = []
    valid_tasks = []

    for task in TASKS:
        task_result = results["results"].get(task, {})
        
        score = None
        metric_name = "unknown"

        if task == "gsm8k":
            # 优先用 flexible-extract（官方推荐）
            if "exact_match,flexible-extract" in task_result:
                score = task_result["exact_match,flexible-extract"]
                metric_name = "exact_match (flexible)"
            elif "exact_match,strict-match" in task_result:
                score = task_result["exact_match,strict-match"]
                metric_name = "exact_match (strict)"
                
        elif task == "mbpp":
            if "pass_at_1,none" in task_result:
                score = task_result["pass_at_1,none"]
                metric_name = "pass@1"
                
        elif task == "ifeval":
            if "prompt_level_strict_acc,none" in task_result:
                score = task_result["prompt_level_strict_acc,none"]
                metric_name = "strict_acc"
                
        elif task in ["xsum", "cnn_dailymail"]:
            if "rouge,none" in task_result:
                rouge_dict = task_result["rouge,none"]
                # 通常用 rouge1 作为代表
                score = rouge_dict.get("rouge1", 0.0)
                metric_name = "rouge1"
                
        else:
            # 通用 fallback（如 acc,none 等）
            for key in ["exact_match,none", "acc,none", "acc_norm,none"]:
                if key in task_result:
                    score = task_result[key]
                    metric_name = key.split(",")[0]
                    break

        if score is None:
            available = list(task_result.keys())
            print(f"[WARN] No standard metric found for {task}. Available: {available}")
            score = 0.0
            metric_name = "N/A"

        score_pct = (score * 100) if isinstance(score, (int, float)) else 0.0
        task_scores.append(score_pct)
        valid_tasks.append(task)
        print(f"{task:15}: {score_pct:6.2f}% ({metric_name})")

    avg_score = sum(task_scores) / len(task_scores)
    print("-" * 60)
    print(f"{'Average':12}: {avg_score:6.2f}%")
    print("="*60)

    # 可选：保存结果到文件
    with open("eval_results.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Average: {avg_score:.2f}\n")
        for name, score in zip(task_names, task_scores):
            f.write(f"{name}: {score:.2f}\n")


if __name__ == "__main__":
    main()