import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import re

def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def concatenate_w1_w3_to_w13(
    base_dir: str,
    rank: int,
    expert_id: int,
    delete_original: bool = False
) -> bool:
    """
    将w1和w3文件拼接成w13文件
    
    Args:
        base_dir: 基础目录（包含experts文件夹）
        rank: 层ID
        expert_id: expert ID
        delete_original: 是否删除原始的w1和w3文件
    
    Returns:
        bool: 是否成功
    """

    w13_key = f"model.layers.{rank}.block_sparse_moe.experts.{expert_id}.w13.weight"
     # 转换为安全的文件名
    w13_safe = sanitize_filename(w13_key)
    # 构建文件路径
    experts_dir = os.path.join(base_dir, "experts")
    w13_file = os.path.join(experts_dir, f"{w13_safe}.safetensors")

    w2_key = f"model.layers.{rank}.block_sparse_moe.experts.{expert_id}.w2.weight"
     # 转换为安全的文件名
    w2_safe = sanitize_filename(w2_key)
    # 构建文件路径
    experts_dir = os.path.join(base_dir, "experts")
    w2_file = os.path.join(experts_dir, f"{w2_safe}.safetensors")
    try:
        # 加载w1权重
        with safe_open(w13_file, framework="pt", device="cpu") as f:
            w13_tensor = f.get_tensor(w13_key)
            print(f"w13形状: {w13_tensor.shape}")
    
        print(w13_tensor)
        with safe_open(w2_file, framework="pt", device="cpu") as f:
            w2_tensor = f.get_tensor(w2_key)
            print(f"w2形状: {w2_tensor.shape}")
    
        print(w2_tensor)
        # # 保存为新的w13文件
        # save_file({w13_key: w13_tensor}, w13_file)
        # print(f"已创建w13文件: {w13_file}")
        # print(f"w13形状: {w13_tensor.shape}")
        # save_file({w2_t_key: w2_tensor}, w2_t_file)
        # print(f"已创建w2文件: {w2_t_file}")
        # print(f"w2形状: {w2_tensor.shape}")
        # # 可选：删除原始文件
        # if delete_original:
        #     os.remove(w1_file)
        #     os.remove(w3_file)
        #     print(f"已删除原始文件: {w1_file}, {w3_file}")
        
        return True
        
    except Exception as e:
        print(f"拼接w1和w3失败: {e}")
        return False

def process_all_experts(
    base_dir: str,
    num_layers: int,
    num_experts_per_layer: int,
    delete_original: bool = False
):
    """
    处理所有层的所有experts
    
    Args:
        base_dir: 基础目录
        num_layers: MoE层数
        num_experts_per_layer: 每层expert数
        delete_original: 是否删除原始文件
    """
    success = concatenate_w1_w3_to_w13(
                base_dir=base_dir,
                rank=23,
                expert_id=39,
                delete_original=delete_original
            )


# 使用示例
if __name__ == "__main__":
    # 1. 预处理：将w1和w3合并为w13
    base_dir = "/sharenvme/usershome/cyl/test/model/Qwen/Qwen1.5-MoE-A2.7B"  # 修改为你的模型目录
    num_layers = 24  # 修改为你的MoE层数
    num_experts_per_layer = 60 # 修改为每层expert数
    
    # 处理所有experts（不删除原始文件）
    process_all_experts(
        base_dir=base_dir,
        num_layers=num_layers,
        num_experts_per_layer=num_experts_per_layer,
        delete_original=False  # 第一次运行时设为False，确认无误后再设为True
    )
    
   