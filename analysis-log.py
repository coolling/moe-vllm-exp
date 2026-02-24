import re
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

def extract_rank_expert_distribution(file_path: str) -> Dict[str, Any]:
    """
    从文件中提取rank和expert分布并计算概率
    
    Args:
        file_path: 文件路径
        
    Returns:
        包含rank专家分布和统计信息的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content:
            return {"error": "文件为空"}
            
    except Exception as e:
        return {"error": f"读取文件失败: {str(e)}"}
    
    # 按"start"分割段
    segments = [seg.strip() for seg in content.split('start') if seg.strip()]
    
    # 正则表达式匹配 rank: X expert: Y 模式
    pattern = r'rank\s*:\s*(\d+)\s*expert\s*:\s*(\d+)'
    
    # 存储统计结果
    rank_expert_counts = defaultdict(Counter)  # rank -> {expert: count}
    total_counts_per_rank = defaultdict(int)   # rank -> 总次数
    
    # 处理所有段
    total_pairs = 0
    for segment in segments:
        matches = re.findall(pattern, segment, re.IGNORECASE)
        
        for rank_str, expert_str in matches:
            try:
                rank = int(rank_str)
                expert = int(expert_str)
                
                rank_expert_counts[rank][expert] += 1
                total_counts_per_rank[rank] += 1
                total_pairs += 1
                
            except ValueError:
                continue
    
    # 如果没有找到数据
    if not rank_expert_counts:
        return {"error": "未找到有效的rank:expert数据"}
    
    # 计算每个rank的专家分布概率并排序
    rank_distributions = {}
    
    for rank in sorted(rank_expert_counts.keys()):
        expert_counter = rank_expert_counts[rank]
        total_for_rank = total_counts_per_rank[rank]
        
        # 计算概率并排序
        distribution = []
        for expert, count in expert_counter.most_common():  # 已经按频率排序
            probability = count / total_for_rank
            
            distribution.append({
                "expert": expert,
                "count": count,
                "probability": round(probability, 6),  # 保留6位小数
                "percentage": round(probability * 100, 4)  # 百分比形式
            })
        
        rank_distributions[str(rank)] = {
            "total_assignments": total_for_rank,
            "unique_experts": len(expert_counter),
            "distribution": distribution,
            "most_common_expert": distribution[0] if distribution else None
        }
    
    # 准备返回结果
    result = {
        "metadata": {
            "file_path": file_path,
            "total_segments": len(segments),
            "total_rank_expert_pairs": total_pairs,
            "unique_ranks": len(rank_distributions),
            "analysis_timestamp": os.path.getmtime(file_path) if os.path.exists(file_path) else None
        },
        "rank_distributions": rank_distributions,
       
    }
    
    return result

def save_distribution_to_json(distribution_data: Dict[str, Any], output_path: str = None) -> str:
    """
    将分布数据保存到JSON文件
    
    Args:
        distribution_data: 分布数据字典
        output_path: 输出文件路径，如果为None则自动生成
        
    Returns:
        保存的文件路径
    """

    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 使用indent和ensure_ascii参数使JSON更易读
            json.dump(distribution_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 数据已保存到: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        return output_path
        
    except Exception as e:
        print(f"✗ 保存JSON文件失败: {str(e)}")
        return ""


def analyze_and_save_distribution(file_path: str, output_json_path: str = None) -> Dict[str, Any]:
    """
    完整的分析并保存流程
    
    Args:
        file_path: 输入文件路径
        output_json_path: 输出JSON文件路径（可选）
        
    Returns:
        分布数据字典
    """
    print(f"开始分析文件: {file_path}")
    print("-" * 40)
    
    # 1. 提取分布数据
    distribution_data = extract_rank_expert_distribution(file_path)
    
    # 2. 检查是否有错误
    if "error" in distribution_data:
        print(f"分析失败: {distribution_data['error']}")
        return distribution_data
    

    
 
    saved_path = save_distribution_to_json(distribution_data, output_json_path)
    

    return distribution_data

# 主要执行函数
def main():
    # 设置文件路径
    dir_t= "/sharenvme/usershome/cyl/test/model/Qwen/Qwen1.5-MoE-A2.7B"
    input_file =dir_t+"/rank_experts_log.txt"
    
    # 可选: 指定输出JSON文件路径
    # output_file = "qwen_rank_expert_distribution.json"
    output_file = dir_t+"/rank_experts_distribution.json"  
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        print("请检查文件路径是否正确")
        return
    
    # 执行分析
    results = analyze_and_save_distribution(input_file, output_file)
    
    # 如果需要直接访问结果数据
    # results变量现在包含了所有分析数据

if __name__ == "__main__":
    main()