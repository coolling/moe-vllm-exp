import re
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Union

def extract_rank_expert_distribution(file_path: str) -> Dict[str, Any]:
    """
    从文件中提取每个段内每个rank的专家分布并计算频率
    
    Args:
        file_path: 文件路径
        
    Returns:
        包含每个段内rank专家分布统计信息的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content:
            return {"error": "文件为空"}
            
    except Exception as e:
        return {"error": f"读取文件失败: {str(e)}"}
    
    # 按"start"分割段
    raw_segments = content.split('start')
    segments = []
    
    # 清理并保留有效段
    for i, seg in enumerate(raw_segments):
        clean_seg = seg.strip()
        if clean_seg:
            segments.append(clean_seg)
    
    # 存储每个段的统计结果
    segment_stats = []
    total_pairs_across_segments = 0
    
    for segment_idx, segment in enumerate(segments):
        # 当前段的统计
        segment_rank_expert_counts = defaultdict(Counter)  # rank -> {expert: count}
        segment_total_counts_per_rank = defaultdict(int)   # rank -> 总次数
        
        # 正则表达式匹配 rank: X expert: Y 模式
        pattern = r'rank\s*:\s*(\d+)\s*expert\s*:\s*(\d+)'
        matches = re.findall(pattern, segment, re.IGNORECASE)
        
        segment_pairs = 0
        for rank_str, expert_str in matches:
            try:
                rank = int(rank_str)
                expert = int(expert_str)
                
                segment_rank_expert_counts[rank][expert] += 1
                segment_total_counts_per_rank[rank] += 1
                segment_pairs += 1
                
            except ValueError:
                continue
        
        total_pairs_across_segments += segment_pairs
        
        # 计算当前段的专家分布频率
        segment_distributions = {}
        
        for rank in sorted(segment_rank_expert_counts.keys()):
            expert_counter = segment_rank_expert_counts[rank]
            total_for_rank = segment_total_counts_per_rank[rank]
            
            # 计算频率并排序
            distribution = []
            for expert, count in expert_counter.most_common():  # 已经按频率排序
                frequency = count / total_for_rank if total_for_rank > 0 else 0
                
                distribution.append({
                    "expert": expert,
                    "count": count,
                    "frequency": round(frequency, 6),  # 保留6位小数
                    "percentage": round(frequency * 100, 4)  # 百分比形式
                })
        
            segment_distributions[str(rank)] = {
                "total_assignments": total_for_rank,
                "unique_experts": len(expert_counter),
                "distribution": distribution,
                "most_frequent_expert": distribution[0] if distribution else None
            }
        
        # 保存当前段的统计
        segment_info = {
            "segment_index": segment_idx,
            "segment_id": f"segment_{segment_idx:03d}",
            "total_pairs_in_segment": segment_pairs,
            "unique_ranks_in_segment": len(segment_distributions),
            "rank_distributions": segment_distributions,
           
        }
        
        segment_stats.append(segment_info)
    
    # 计算跨所有段的汇总统计
    all_rank_expert_counts = defaultdict(Counter)
    all_total_counts_per_rank = defaultdict(int)
    
    for segment_info in segment_stats:
        for rank_str, rank_info in segment_info["rank_distributions"].items():
            rank = int(rank_str)
            for expert_dist in rank_info["distribution"]:
                expert = expert_dist["expert"]
                count = expert_dist["count"]
                all_rank_expert_counts[rank][expert] += count
                all_total_counts_per_rank[rank] += count
    
    # 计算整体分布
    overall_distributions = {}
    for rank in sorted(all_rank_expert_counts.keys()):
        expert_counter = all_rank_expert_counts[rank]
        total_for_rank = all_total_counts_per_rank[rank]
        
        distribution = []
        for expert, count in expert_counter.most_common():
            frequency = count / total_for_rank if total_for_rank > 0 else 0
            
            distribution.append({
                "expert": expert,
                "count": count,
                "frequency": round(frequency, 6),
                "percentage": round(frequency * 100, 4)
            })
        
        overall_distributions[str(rank)] = {
            "total_assignments": total_for_rank,
            "unique_experts": len(expert_counter),
            "distribution": distribution,
            "most_frequent_expert": distribution[0] if distribution else None
        }
    
    # 准备返回结果
    result = {
        "metadata": {
            "file_path": file_path,
            "total_segments": len(segments),
            "total_segments_analyzed": len(segment_stats),
            "total_rank_expert_pairs": total_pairs_across_segments,
            "unique_ranks_overall": len(overall_distributions),
            "analysis_timestamp": os.path.getmtime(file_path) if os.path.exists(file_path) else None,
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        },
        "overall_statistics": {
            "rank_distributions": overall_distributions,
            "summary_by_rank": {
                rank_str: {
                    "total_assignments": info["total_assignments"],
                    "unique_experts": info["unique_experts"],
                    "top_expert": info["most_frequent_expert"]["expert"] if info["most_frequent_expert"] else None,
                    "top_expert_frequency": info["most_frequent_expert"]["frequency"] if info["most_frequent_expert"] else None
                }
                for rank_str, info in overall_distributions.items()
            }
        },
        "segment_analysis": segment_stats,
        # 方便快速访问的索引
        "indices": {
            "segment_by_index": {info["segment_index"]: info["segment_id"] for info in segment_stats},
            "ranks_present": sorted([int(r) for r in overall_distributions.keys()])
        }
    }
    
    return result


def print_segment_statistics(distribution_data: Dict[str, Any], max_segments: int = 5) -> None:
    """
    打印每个段的统计信息
    
    Args:
        distribution_data: 分布数据字典
        max_segments: 最多打印的段数
    """
    if "error" in distribution_data:
        print(f"错误: {distribution_data['error']}")
        return
    
    segment_stats = distribution_data.get("segment_analysis", [])
    
    print(f"\n{'='*60}")
    print(f"段分析统计 (共 {len(segment_stats)} 个段)")
    print(f"{'='*60}")
    
    # 打印前几个段的详细统计
    for i, segment_info in enumerate(segment_stats[:max_segments]):
        print(f"\n[段 {segment_info['segment_index']}] {segment_info['segment_id']}")
        print(f"  总匹配数: {segment_info['total_pairs_in_segment']}")
        print(f"  唯一rank数: {segment_info['unique_ranks_in_segment']}")
        
        for rank_str, rank_info in segment_info["rank_distributions"].items():
            top_expert = rank_info["most_frequent_expert"]
            if top_expert:
                print(f"  Rank {rank_str}: 总分配 {rank_info['total_assignments']} 次, "
                      f"最频繁专家 {top_expert['expert']} (频率: {top_expert['percentage']:.2f}%)")
    
    if len(segment_stats) > max_segments:
        print(f"\n... 还有 {len(segment_stats) - max_segments} 个段未显示")


def compare_segments(distribution_data: Dict[str, Any], rank_to_compare: int = None) -> Dict[str, Any]:
    """
    比较不同段之间的专家频率分布
    
    Args:
        distribution_data: 分布数据字典
        rank_to_compare: 要比较的特定rank（为None时比较所有rank）
        
    Returns:
        比较结果字典
    """
    if "error" in distribution_data:
        return {"error": distribution_data["error"]}
    
    segment_stats = distribution_data.get("segment_analysis", [])
    comparison = {}
    
    if rank_to_compare is not None:
        # 比较特定rank在不同段中的分布
        rank_str = str(rank_to_compare)
        rank_comparison = []
        
        for segment_info in segment_stats:
            if rank_str in segment_info["rank_distributions"]:
                rank_info = segment_info["rank_distributions"][rank_str]
                top_expert = rank_info["most_frequent_expert"]
                
                rank_comparison.append({
                    "segment_id": segment_info["segment_id"],
                    "segment_index": segment_info["segment_index"],
                    "total_assignments": rank_info["total_assignments"],
                    "unique_experts": rank_info["unique_experts"],
                    "top_expert": top_expert["expert"] if top_expert else None,
                    "top_expert_frequency": top_expert["frequency"] if top_expert else None,
                    "expert_diversity": len(rank_info["distribution"])
                })
        
        comparison[f"rank_{rank_to_compare}"] = {
            "total_segments_with_rank": len(rank_comparison),
            "segments": rank_comparison,
            "consistency": {
                "avg_assignments": sum(r["total_assignments"] for r in rank_comparison) / len(rank_comparison) if rank_comparison else 0,
                "most_common_top_expert": Counter(r["top_expert"] for r in rank_comparison).most_common(1)[0] if rank_comparison else None
            }
        }
    
    return comparison


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
        print(f"  文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        return output_path
        
    except Exception as e:
        print(f"✗ 保存JSON文件失败: {str(e)}")
        return ""


def analyze_and_save_distribution(file_path: str, output_json_path: str = None, 
                                  verbose: bool = True) -> Dict[str, Any]:
    """
    完整的分析并保存流程
    
    Args:
        file_path: 输入文件路径
        output_json_path: 输出JSON文件路径（可选）
        verbose: 是否显示详细输出
        
    Returns:
        分布数据字典
    """
    if verbose:
        print(f"开始分析文件: {file_path}")
        print(f"文件大小: {os.path.getsize(file_path) / 1024:.2f} KB" if os.path.exists(file_path) else "文件不存在")
        print("-" * 40)
    
    # 1. 提取分布数据
    distribution_data = extract_rank_expert_distribution(file_path)
    
    # 2. 检查是否有错误
    if "error" in distribution_data:
        if verbose:
            print(f"分析失败: {distribution_data['error']}")
        return distribution_data
    
    # 3. 显示基本统计
    if verbose:
        metadata = distribution_data["metadata"]
        print(f"✓ 分析完成!")
        print(f"  总段数: {metadata['total_segments']}")
        print(f"  总匹配对: {metadata['total_rank_expert_pairs']}")
        print(f"  唯一rank数: {metadata['unique_ranks_overall']}")
        
        # 打印每个段的简要统计
        print_segment_statistics(distribution_data, max_segments=10)
    
    # 4. 保存到JSON
    saved_path = save_distribution_to_json(distribution_data, output_json_path)
    
    if verbose and saved_path:
        print(f"\n✓ 详细分析结果已保存到: {saved_path}")
    
    return distribution_data


def generate_summary_report(distribution_data: Dict[str, Any], 
                           output_report_path: str = None) -> str:
    """
    生成文本格式的汇总报告
    
    Args:
        distribution_data: 分布数据字典
        output_report_path: 输出报告文件路径
        
    Returns:
        报告文件路径
    """
    if "error" in distribution_data:
        return ""
    
    if output_report_path is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_report_path = f"rank_expert_summary_{timestamp}.txt"
    
    try:
        with open(output_report_path, 'w', encoding='utf-8') as f:
            metadata = distribution_data["metadata"]
            f.write(f"{'='*60}\n")
            f.write(f"Rank-Expert 频率分析报告\n")
            f.write(f"生成时间: {metadata.get('analysis_timestamp', 'N/A')}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"文件信息:\n")
            f.write(f"  文件路径: {metadata['file_path']}\n")
            f.write(f"  总段数: {metadata['total_segments']}\n")
            f.write(f"  总匹配对: {metadata['total_rank_expert_pairs']}\n")
            f.write(f"  唯一rank数: {metadata['unique_ranks_overall']}\n\n")
            
            # 整体统计
            overall = distribution_data["overall_statistics"]
            f.write(f"{'='*60}\n")
            f.write(f"整体统计\n")
            f.write(f"{'='*60}\n")
            
            for rank_str, rank_info in overall["rank_distributions"].items():
                top_expert = rank_info["most_frequent_expert"]
                if top_expert:
                    f.write(f"Rank {rank_str}:\n")
                    f.write(f"  总分配次数: {rank_info['total_assignments']}\n")
                    f.write(f"  唯一专家数: {rank_info['unique_experts']}\n")
                    f.write(f"  最频繁专家: {top_expert['expert']} (频率: {top_expert['percentage']:.2f}%)\n\n")
            
            # 段级统计摘要
            segment_stats = distribution_data["segment_analysis"]
            f.write(f"{'='*60}\n")
            f.write(f"段级统计摘要 (共 {len(segment_stats)} 个段)\n")
            f.write(f"{'='*60}\n")
            
            for segment_info in segment_stats[:10]:  # 只显示前10个段
                f.write(f"\n段 {segment_info['segment_index']}:\n")
                f.write(f"  匹配对数: {segment_info['total_pairs_in_segment']}\n")
                f.write(f"  唯一rank数: {segment_info['unique_ranks_in_segment']}\n")
        
        print(f"✓ 汇总报告已生成: {output_report_path}")
        return output_report_path
        
    except Exception as e:
        print(f"✗ 生成报告失败: {str(e)}")
        return ""


# 主要执行函数
def main():
    # 设置文件路径
    dir_t = "/sharenvme/usershome/cyl/test/model/mistralai/Mixtral-8x7B-Instruct-v0.1"
    input_file = os.path.join(dir_t, "rank_experts_log.txt")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        print("请检查文件路径是否正确")
        return
    
    # 可选: 指定输出JSON文件路径
    output_file = os.path.join(dir_t, "segment_freq_analysis.json")
    
    # 执行分析（verbose=True显示详细输出）
    results = analyze_and_save_distribution(input_file, output_file, verbose=True)
    
    # 如果分析成功，可以进一步操作
    if "error" not in results:
        # 1. 生成文本报告
        report_file = os.path.join(dir_t, "segment_analysis_summary.txt")
        generate_summary_report(results, report_file)
        
        # 2. 比较特定rank在不同段中的分布
        print(f"\n{'='*60}")
        print("比较不同段中Rank 0的专家分布:")
        print('='*60)
        comparison = compare_segments(results, rank_to_compare=0)
        if "rank_0" in comparison:
            rank_data = comparison["rank_0"]
            print(f"Rank 0出现在 {rank_data['total_segments_with_rank']} 个段中")
            if rank_data["segments"]:
                top_expert_counter = Counter(s["top_expert"] for s in rank_data["segments"])
                most_common, count = top_expert_counter.most_common(1)[0]
                print(f"最常出现的最频繁专家: {most_common} (在 {count} 个段中)")
        
        # 3. 可以访问具体数据
        print(f"\n{'='*60}")
        print("数据访问示例:")
        print('='*60)
        print(f"总段数: {results['metadata']['total_segments']}")
        
        # 访问第一个段的详细信息
        if results["segment_analysis"]:
            first_segment = results["segment_analysis"][0]
            print(f"第一个段 (索引 {first_segment['segment_index']}):")
            print(f"  匹配对数: {first_segment['total_pairs_in_segment']}")
            
            if first_segment["rank_distributions"]:
                first_rank = list(first_segment["rank_distributions"].keys())[0]
                rank_info = first_segment["rank_distributions"][first_rank]
                print(f"  Rank {first_rank} 有 {rank_info['unique_experts']} 个不同专家")


if __name__ == "__main__":
    main()