import re
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
# import matplotlib.pyplot as plt
import numpy as np
t=2
def plot_basic_bar(sorted_items, top_n=200):
    """
    绘制基础的条形图
    """
    # 限制显示数量
    display_items = sorted_items[:top_n]
    
    # 提取标签和值
    labels = []
    values = []
    
    for pattern, count in display_items:
        # 简化标签：如果是元组，转换为字符串，限制长度
        if isinstance(pattern, tuple) and len(pattern) > 3:
            label = f"{len(pattern)}"
        else:
            label = str(pattern)[:30] + "..." if len(str(pattern)) > 30 else str(pattern)
        labels.append(label)
        values.append(count)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建颜色渐变
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(values)))
    
    # 绘制条形图
    bars = plt.bar(range(len(values)), values, color=colors, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    # 设置x轴标签
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 添加标题和标签
    plt.title(f'(Top {len(display_items)})', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('t', fontsize=12, labelpad=10)
    plt.ylabel('c', fontsize=12, labelpad=10)
    
    # 添加网格
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    # 保存图片（可选）
    plt.savefig('pattern_distribution.png', dpi=300, bbox_inches='tight')
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
    print(len(segments))
    # return
    # 存储统计结果

    
    # 处理所有段
    total_pairs = 0
    res=defaultdict(int)
    
    re_t=[]
    is_start=True
    for segment in segments:
        matches = re.findall(pattern, segment, re.IGNORECASE)
        
        for rank_str, expert_str in matches:
            try:
                rank = int(rank_str)
                expert = int(expert_str)
                
                if rank==0 :
                    if is_start:
                        re_t.append((rank,expert))
                    else:
                        is_start=True
                        # res.append(re_t)
                        
                        if(len(re_t)>96):
                            qqq=1
                        else:
                            print(re_t)
                        
                            res[tuple(re_t)]+=1
                            total_pairs += 1
                        # print(res)
                        # return
                        re_t=[]
                        re_t.append((rank,expert))
                else:
                    re_t.append((rank,expert))
                    is_start=False
            except ValueError:
                continue


    # print(len(res)-c)
    print(total_pairs)
    # sorted_items = sorted(res.items(), key=lambda x: x[1], reverse=True)
    # print("按值从大到小排序，取前10:")
    # plot_basic_bar(sorted_items)

    # import time
    # start=time.time()
    pattern_counter = defaultdict(Counter)
    print(len(res))
    ci=0
    for key ,value in res.items():
        rank_groups = defaultdict(list)
        for rank, expert in key:
            rank_groups[rank].append(expert)
        sorted_ranks = sorted(rank_groups.keys())
        for i in range(len(sorted_ranks) - t):
            key_parts = []
            for j in range(t):
                current_rank = sorted_ranks[i + j]
                key_parts.append(current_rank)
                key_parts.append(tuple(rank_groups[current_rank]))
            key = tuple(key_parts)
            rank6 = sorted_ranks[i + t]
            for exp in rank_groups[rank6]:
                pattern_counter[str(key)][exp]+=value
                if pattern_counter[str(key)][exp]==1:
                    ci+=1
    print("ci",ci)
            # return
    # elapsed_ms = (time.time() - start) * 1000
    # print(f"[manager.release_current_layer() {elapsed_ms:.2f} ms")
    sorted_pattern_counter = {}
    # for pattern, counter in pattern_counter.items():
    #     # 使用 most_common() 方法直接获得排序后的列表
    #     sorted_items = counter.most_common()  # 默认按值降序
    #     sorted_pattern_counter[pattern] = dict(sorted_items)
    #     print(sorted_pattern_counter[pattern])
    sorted_patterns = sorted(
        pattern_counter.items(), 
        key=lambda x: (len(x[1]), sum(x[1].values())), 
        reverse=True
    )

    # 此时 sorted_patterns 是一个 list，元素为 (pattern, counter)
    ccc=0
    for pattern, counter in sorted_patterns:
        # 内部依然按你之前的逻辑：专家按出现频次降序
        sorted_items = counter.most_common()
        sorted_pattern_counter[pattern] = dict(sorted_items)
        
        # 打印结果查看排序效果
        print(f"Pattern: {pattern}")
        print(f"专家种类数: {len(counter)}, 总激活次数: {sum(counter.values())}")
        print(f"分布详情: {sorted_pattern_counter[pattern]}\n")
        ccc+=1
        if ccc >10 :
            break
    # k=(0, (2, 3), 1, (0, 1))
    # print(sorted_pattern_counter[str(k)])
    return sorted_pattern_counter

def save_distribution_to_json(distribution_data: Dict[str, Any], output_path: str = None) -> str:
    """
    将分布数据保存到JSON文件
    
    Args:
        distribution_data: 分布数据字典
        output_path: 输出文件路径，如果为None则自动生成
        
    Returns:
        保存的文件路径
    """
    if output_path is None:
        # 自动生成文件名
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"rank_expert_segment_analysis_{timestamp}.json"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 使用indent和ensure_ascii参数使JSON更易读
            json.dump(distribution_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 数据已保存到: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024/1024:.2f} MB")
        
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
    
    saved_path = save_distribution_to_json(distribution_data, output_json_path)
    

    print(f"\n✓ 详细分析结果已保存到: {saved_path}")
    
    return distribution_data

# 主要执行函数
def main():
    # 设置文件路径
    # dir_t= "/sharenvme/usershome/cyl/test/model/mistralai/Mixtral-8x7B-Instruct-v0.1"
    dir_t="/sharenvme/usershome/cyl/test/model/Qwen/Qwen1.5-MoE-A2.7B"
    input_file =dir_t+"/rank_experts_log.txt"
    
    # 可选: 指定输出JSON文件路径
    # output_file = "qwen_rank_expert_distribution.json"
    output_file = dir_t+"/route_experts_distribution-"+str(t)+".json"  
    
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