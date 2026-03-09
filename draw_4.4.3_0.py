import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d  # 【新增】引入高斯滤波用于极致平滑

def process_txt_file(file_path):
    length_time_dict = defaultdict(list)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                length = int(float(parts[0]))
                if length > 300 or length < 15:
                    continue
                time_val = float(parts[1])
                length_time_dict[length].append(time_val)
        
        length_avg_time = []
        for length, time_list in length_time_dict.items():
            avg_time = sum(time_list) / len(time_list)
            # 将计算出的总耗时除以 1000，把毫秒(ms)转换为秒(s)
            length_avg_time.append((length, (avg_time * length) / 1000.0))
        
        length_avg_time.sort(key=lambda x: x[0])
        sorted_lengths = [item[0] for item in length_avg_time]
        sorted_avg_times = [item[1] for item in length_avg_time]
        
        return sorted_lengths, sorted_avg_times
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出现异常：{str(e)}")
        return [], []

def smooth_curve(x, y, num_points=500, sigma=8):
    """
    使用三次样条插值 + 高斯滤波 生成极致平滑曲线
    :param sigma: 平滑度控制参数，值越大曲线越平滑（抹平震荡），默认设为 12
    """
    if len(x) < 4:
        return x, y
    
    x_array = np.array(x)
    y_array = np.array(y)
    
    # 1. 生成极其密集的 X 轴数据点
    x_smooth = np.linspace(x_array.min(), x_array.max(), num_points)
    
    # 2. 先进行基础的三次样条插值
    spline = make_interp_spline(x_array, y_array, k=3)
    y_interp = spline(x_smooth)
    
    # 3. 【核心修改】对插值后的数据进行高斯滤波，消除系统噪声带来的锯齿和波浪
    y_smooth = gaussian_filter1d(y_interp, sigma=sigma)
    
    return x_smooth, y_smooth

if __name__ == "__main__":
    # -------------------------- 核心样式重置 --------------------------
    plt.rcParams.update({'font.size': 22, 'font.family': 'serif'})
    
    file_paths = [

          "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_0_m.txt",
        "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_1_m.txt",
        "/sharenvme/usershome/cyl/test/prefill_data/prefill_moevllm_cache_2_m.txt",
        "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_3_m.txt",
        "/sharenvme/usershome/cyl/test/prefill_data/prefill_cache_4_m.txt",
        
    ]
    
    # 【修复】补充 Qwen 文件的映射，确保图例显示为 Baseline 名称
    label_map = {
        "prefill_cache_0_m.txt": "Cache Size = 0",
        "prefill_cache_1_m.txt": "Cache Size = 1",
        "prefill_moevllm_cache_2_m.txt": "Cache Size = 2 (Ours)",
        "prefill_cache_3_m.txt": "Cache Size = 3",
        "prefill_cache_4_m.txt": "Cache Size = 4"
    }
    
    # 你要求的定制颜色
    custom_orange = (251/255, 125/255, 14/255)
    custom_blue = (0/255, 191/255, 255/255)
    
    # 颜色配置
    colors = ['#555555', '#d62728', '#2ca02c', custom_blue, custom_orange]
    
    all_data = []
    for file_path in file_paths:
        lengths, avg_times = process_txt_file(file_path)
        all_data.append((file_path, lengths, avg_times))
    
    has_valid_data = all(len(lengths) > 0 and len(avg_times) > 0 for _, lengths, avg_times in all_data)
    
    if has_valid_data:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        
        for idx, (file_path, lengths, avg_times) in enumerate(all_data):
            file_name = file_path.split('/')[-1]
            label_name = label_map.get(file_name, file_name)
            
            # 1. 获取平滑后的数据用于画线
            x_smooth, y_smooth = smooth_curve(lengths, avg_times) # 觉得不够平滑可以把 sigma 改到 15 或 20
            
            # 2. 画平滑的趋势线
            ax.plot(x_smooth, y_smooth, linewidth=2.5, label=label_name, color=colors[idx], alpha=0.9)
            
        # 坐标轴设置
        ax.set_xlabel('Prompt Length (# Tokens)', fontsize=24)
        ax.set_ylabel('Prefill Latency (s)', fontsize=26)
        ax.set_ylim(5, 55)

        # 例如：强制 Y 轴从 0 开始，最大值设为 15
        
        # 图例设置
        ax.legend(fontsize=17, frameon=False, loc='upper left')
        
        fig.tight_layout()
        plt.savefig('prefill_latency_smoothed_4.4.3.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('prefill_latency_smoothed_4.4.3.png', bbox_inches='tight')
        print("平滑绘图成功！已保存为 prefill_latency_smoothed.pdf 和 prefill_latency_smoothed.png")
    else:
        print("无法绘图：未读取到有效数据")
        
     