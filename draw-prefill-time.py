import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.font_manager as fm
from scipy.interpolate import make_interp_spline  # 导入插值平滑工具（需确保scipy已安装）


def process_txt_file(file_path):
    """
    处理单个txt文件，聚合相同长度的时间（取平均），并按长度排序
    :param file_path: txt文件路径（字符串）
    :return: 排序后的长度列表、对应平均时间列表
    """
    # 用字典存储相同长度对应的所有时间，key=长度，value=时间列表
    length_time_dict = defaultdict(list)
    
    try:
        # 读取txt文件，跳过可能的空行，处理数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # 去除行首尾空格、换行符
                if not line:  # 跳过空行
                    continue
                # 分割数据（按空格分割，适配多空格分隔场景）
                parts = line.split()
                if len(parts) != 2:  # 跳过格式错误的行（非两列数据）
                    print(f"警告：文件{file_path}中，行'{line}'格式错误，需两列数据，已跳过")
                    continue
                # 转换为数值类型（长度转为整数，时间转为浮点数）
                length = int(float(parts[0]))  # 兼容可能的浮点型长度（如10.0）
                if length>100:
                    continue
                if length<15:
                    continue
                time_val = float(parts[1])
                # 将时间添加到对应长度的列表中
                length_time_dict[length].append(time_val)
        
        # 计算每个长度对应的平均时间
        length_avg_time = []
        for length, time_list in length_time_dict.items():
            avg_time = sum(time_list) / len(time_list)
            length_avg_time.append( (length, avg_time*length) )
        
        # 按长度从小到大排序
        length_avg_time.sort(key=lambda x: x[0])
        # 拆分排序后的长度和平均时间，用于绘图
        sorted_lengths = [item[0] for item in length_avg_time]
        sorted_avg_times = [item[1] for item in length_avg_time]
        
        return sorted_lengths, sorted_avg_times
    
    except FileNotFoundError:
        print(f"错误：文件{file_path}未找到，请检查文件路径是否正确")
        return [], []
    except Exception as e:
        print(f"处理文件{file_path}时出现异常：{str(e)}")
        return [], []

# 新增：曲线平滑函数（适配离散数据，避免过度平滑导致失真）
def smooth_curve(x, y, smooth_factor=0.35):
    return x,y
    """
    对离散的x、y数据进行平滑处理，采用三次样条插值
    修复：移除make_interp_spline不兼容的`s`参数，改用默认插值+密度调整实现平滑
    :param x: 原始x轴数据（长度列表）
    :param y: 原始y轴数据（平均时间列表）
    :param smooth_factor: 平滑因子（0~1，越小越贴合原始数据，越大越平滑）
    :return: 平滑后的x、y数据
    """
    # 确保数据量足够进行平滑（至少3个数据点）
    if len(x) < 3:
        return x, y  # 数据点过少，不进行平滑，返回原始数据
    # 生成更密集的x轴点（平滑因子越小，点数越少，越贴合原始数据；反之越平滑）
    x_smooth = np.linspace(min(x), max(x), int(300 * (1 - smooth_factor) + 100))
    # 三次样条插值（移除s参数，适配所有scipy版本，避免TypeError）
    spline = make_interp_spline(x, y)
    # 生成平滑后的y轴数据
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# -------------------------- 核心执行部分 --------------------------
if __name__ == "__main__":
    # 解决Linux环境字体报错问题（适配系统默认字体，无需额外安装）
    # plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题
    
    # 1. 输入文件路径列表（支持任意数量txt文件，可直接添加/删除路径，适配多条曲线绘制）
    file_paths = [
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_log_nocache_sort_maxsize=2_m.txt","/sharenvme/usershome/cyl/test/prefill_data/prefill_log_nocache_sort_re_maxsize=1_m.txt.txt",
        
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_layerwise_m.txt",  # 第一条曲线对应文件
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_ondemand_m.txt"   ,  # 第二条曲线对应文件
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_moevllm_cache_2_m.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_init_m.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_m.txt",
        # "/sharenvme/usershome/cyl/test/prefill_log.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_0_m.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_1_m.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_3_m.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_4_m.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_5_m.txt",
        
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_ondemand_q.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_layerwise_q.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_q.txt",
        # "/sharenvme/usershome/cyl/test/prefill_data/prefill_moevllm_cache_15_q.txt"
        
        "/sharenvme/usershome/cyl/test/prefill_data/prefill_layerwise_q.txt",
        "/sharenvme/usershome/cyl/test/prefill_data/重排实验：prefill_log_nocache_sort_maxsize=2_q.txt",
    ]
    
    # 2. 批量处理所有txt文件，存储每条曲线的长度和平均时间数据
    all_data = []  # 列表元素为元组：(文件路径, 长度列表, 平均时间列表)
    for file_path in file_paths:
        lengths, avg_times = process_txt_file(file_path)
        all_data.append( (file_path, lengths, avg_times) )
    
    # 3. 绘制多条曲线（确保所有文件都读取到有效数据才绘图）
    # 校验所有文件是否都有有效数据（0-200范围内的长度数据）
    has_valid_data = all( len(lengths) > 0 and len(avg_times) > 0 for _, lengths, avg_times in all_data )
    
    if has_valid_data:
        # 设置绘图样式，提升可读性
        plt.figure(figsize=(10, 6), dpi=100)
        
        # 定义曲线颜色列表（可根据文件数量添加颜色，避免颜色重复）
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 批量绘制每条曲线，标注清晰（按文件列表顺序绘制）
        for idx, (file_path, lengths, avg_times) in enumerate(all_data):
            # 循环取色，超出颜色列表长度时重新从开头取色
            color = colors[idx % len(colors)]
            # 绘制曲线，标注文件路径（简化标签，仅保留文件名，更简洁）
            file_name = file_path.split('/')[-1]  # 提取文件名（如prefill_layerwise_m.txt）
            # 核心修改：对当前曲线数据进行平滑处理
            x_smooth, y_smooth = smooth_curve(lengths, avg_times)
            # 绘制平滑后的曲线（保留原线条样式，提升美观度）
            plt.plot(x_smooth, y_smooth, linewidth=2, label=f'{file_name}', color=color)
        
        # 设置坐标轴标签、标题和图例（适配英文，避免中文字体报错）
        plt.xlabel('Length', fontsize=12)
        plt.ylabel('Average Time', fontsize=12)
        plt.title('Relationship Curve between Length and Corresponding Average Time in Multiple Txt Files', fontsize=14, pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)  # 添加网格，便于查看数据
    
        
        # 调整布局，避免标签被截断
        # plt.tight_layout()
        # 保存图像（覆盖原有保存逻辑，直接保存为curve.png）
        plt.savefig('curve.png')
    else:
        # 提示具体哪些文件无有效数据，便于排查
        invalid_files = [file_path for file_path, lengths, avg_times in all_data if len(lengths) == 0 or len(avg_times) == 0]
        print(f"无法绘图：以下文件未读取到有效数据（需包含0-200范围内的长度数据），请检查文件内容和路径：")
        for file in invalid_files:
            print(f" - {file}")