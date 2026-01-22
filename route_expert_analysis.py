import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict

class RouteExpertAnalyzer:
    """Route-Expert分布分析器"""
    
    def __init__(self, json_file_path: str):
        """
        初始化分析器
        
        Args:
            json_file_path: JSON文件路径
        """
        self.file_path = json_file_path

        self.route_distributions = None
        
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
                data = json.load(f)
            
            # 提取主要部分
       
            self.route_distributions = data

            return True
            
        except json.JSONDecodeError as e:
            print(f"错误: JSON文件格式错误 - {e}")
            return False
        except Exception as e:
            print(f"错误: 加载文件失败 - {e}")
            return False

    
    def get_route_distribution(self, key) -> Optional[Dict[str, Any]]:

        key_str = str(key)
        if key_str in self.route_distributions:
            return self.route_distributions[key_str]
        return None
    
  
    
    
    


# 直接使用的示例
def quick_analysis_example():

    json_file = "/sharenvme/usershome/cyl/test/route_experts_distribution.json"

    
    
    # 创建分析器
    analyzer = RouteExpertAnalyzer(json_file)
    print(analyzer.get_route_distribution((0, (2, 3), 1, (0, 1))))

   


if __name__ == "__main__":
    quick_analysis_example()