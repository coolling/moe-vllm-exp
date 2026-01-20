import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict

class RankExpertAnalyzer:
    """Rank-Expert分布分析器"""
    
    def __init__(self, json_file_path: str):
        """
        初始化分析器
        
        Args:
            json_file_path: JSON文件路径
        """
        self.file_path = json_file_path

        self.rank_distributions = None
        
        # 加载数据
        self.load_data()
    
    def load_data(self) -> bool:
        """
        加载JSON数据
        
        Returns:
            成功加载返回True，否则返回False
        """
        try:
            if not os.path.exists(self.file_path):
                print(f"错误: 文件不存在 - {self.file_path}")
                return False
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # 提取主要部分
       
            self.rank_distributions = self.data.get('rank_distributions', {})

            return True
            
        except json.JSONDecodeError as e:
            print(f"错误: JSON文件格式错误 - {e}")
            return False
        except Exception as e:
            print(f"错误: 加载文件失败 - {e}")
            return False
    

    

    
    def get_rank_distribution(self, rank: int) -> Optional[Dict[str, Any]]:
        """
        获取特定rank的分布信息
        
        Args:
            rank: 要查询的rank编号
            
        Returns:
            分布信息字典，如果不存在则返回None
        """
        rank_str = str(rank)
        if rank_str in self.rank_distributions:
            return self.rank_distributions[rank_str]
        return None
    
  
    
    
    

def predict_experts_order(analyzer,rank):
        re=[]
        if analyzer.get_rank_distribution(rank):
            for exp_info in analyzer.get_rank_distribution(rank)['distribution']:
                re.append(exp_info['expert'])
        for i in range(4):
            if i not in re:
                re.append(i)
        return re

# 直接使用的示例
def quick_analysis_example():
    """快速分析示例"""
    # 假设你的JSON文件名为 'qwen_sharegpt_log_rank_expert_distribution.json'
    json_file = "/sharenvme/usershome/cyl/test/model/Isotonic/smol_llama-4x220M-MoE/rank_experts_distribution.json"
    st_file_dir = os.path.dirname(json_file )
    if not os.path.exists(json_file):
        print(f"文件 {json_file} 不存在")
    
    
    # 创建分析器
    analyzer = RankExpertAnalyzer(json_file)
    print(analyzer.get_rank_distribution(2))
    print(st_file_dir)
    print(predict_experts_order(analyzer,2))
   


if __name__ == "__main__":
    quick_analysis_example()