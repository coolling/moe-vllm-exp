#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# ----------------------------
# 配置
# ----------------------------
MODEL_NAME = "./model/Isotonic/smol_llama-4x220M-MoE"
DATASET_PATH = "./dataset/pacovaldez/stackoverflow-questions/data/post_questions_test/post_questions_test_000000000000.parquet"  # 数据集路径
TARGET_LAYER = 5  # 要分析的 MoE 层（从 0 开始）

# ----------------------------
# 主分析函数：分析单条输入的专家激活（返回激活计数）
# ----------------------------
def analyze_input_expert_usage(model, tokenizer, text: str, target_layer: int):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, output_router_logits=True)

    router_logits = outputs.router_logits
    if isinstance(router_logits, (tuple, list)):
        router_list = list(router_logits)
    elif hasattr(router_logits, 'shape') and len(router_logits.shape) == 4:
        router_list = [router_logits[i] for i in range(router_logits.shape[0])]
    else:
        raise ValueError(f"Unsupported router_logits type: {type(router_logits)}")

    num_layers = len(router_list)
    if target_layer < 0:
        target_layer += num_layers
    if not (0 <= target_layer < num_layers):
        raise ValueError(f"target_layer={target_layer} out of range for {num_layers} layers")

    logits = router_list[target_layer]
    if logits.dim() == 3:
        logits = logits.squeeze(0)  # [seq_len, num_experts]

    assert logits.shape[0] == seq_len, f"Logits seq_len {logits.shape[0]} != input seq_len {seq_len}"

    num_experts = logits.shape[-1]
    config = model.config
    experts_per_tok = getattr(config, "num_experts_per_tok", 2)

    total_activations = defaultdict(int)
    topk_indices = torch.topk(logits, k=experts_per_tok, dim=-1).indices  # [seq_len, K]
    for expert_ids in topk_indices:
        for eid in expert_ids.tolist():
            total_activations[eid] += 1

    return total_activations, seq_len


# ----------------------------
# 主流程：遍历整个数据集，累计专家激活
# ----------------------------
def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        output_router_logits=True,
    )
    model.eval()

    # 全局累计
    global_activations = defaultdict(int)
    total_tokens_processed = 0
    dialog_count = 0

    print(f"\nReading dataset from: {DATASET_PATH}")
    try:
        df = pd.read_parquet(DATASET_PATH)
        for line_num, line in df.iterrows():
                if line_num>50:
                    break
                if line.empty:
                    continue
                try:
                    # print(line)
                    context = line["title"]
                    
                    response = line["body"]

                    if not context or not response:
                        print(f"⚠️  Skipping line {line_num}: missing Context or Response")
                        continue

                    # 拼接为模型可理解的格式（可根据训练格式调整）
                    dialogue = f"User: {context}\nAgent: {response}"

                    # 分析当前对话
                    activations, seq_len = analyze_input_expert_usage(
                        model, tokenizer, dialogue, target_layer=TARGET_LAYER
                    )

                    # 累加到全局统计
                    for eid, count in activations.items():
                        print(line_num,eid,count)
                        global_activations[eid] += count
                    total_tokens_processed += seq_len
                    dialog_count += 1

                    if dialog_count % 10 == 0:
                        print(f"Processed {dialog_count} dialogs...")

                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error processing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        return

    # 最终统计
    if not global_activations:
        print("No valid dialogues processed.")
        return

    total_selections = sum(global_activations.values())
    expected_per_token = getattr(model.config, "num_experts_per_tok", 2)
    expected_total = expected_per_token * total_tokens_processed

    print("\n" + "="*60)
    print(f"✅ Total processed: {dialog_count} dialogs, {total_tokens_processed} tokens")
    print(f"Total expert selections: {total_selections} (expected: {expected_total})")

    # 计算频率
    expert_freq = {eid: count / total_selections for eid, count in global_activations.items()}

    print(f"\n📊 Final Expert Activation Frequency (Layer {TARGET_LAYER}):")
    for eid in sorted(expert_freq.keys()):
        count = global_activations[eid]
        freq = expert_freq[eid]
        print(f"  Expert {eid:2d}: {count:6d} times ({freq:.2%})")

    if abs(total_selections - expected_total) > 1e-5:
        print("⚠️  Warning: Total selections ≠ expected. Check routing logic.")

if __name__ == "__main__":
    main()