import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d 

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
                # 过滤掉过长或过短的异常 Prompt 长度
                if length > 230 or length < 15:
                    continue
                time_val = float(parts[1])
                length_time_dict[length].append(time_val)
        
        length_avg_time = []
        for length, time_list in length_time_dict.items():
            avg_time = sum(time_list) / len(time_list)
            # 将计算出的总耗时除以 1000，转换为秒(s)
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
    """
    if len(x) < 4:
        return x, y
    
    x_array = np.array(x)
    y_array = np.array(y)
    
    x_smooth = np.linspace(x_array.min(), x_array.max(), num_points)
    spline = make_interp_spline(x_array, y_array, k=3)
    y_interp = spline(x_smooth)
    y_smooth = gaussian_filter1d(y_interp, sigma=sigma)
    
    return x_smooth, y_smooth

if __name__ == "__main__":
    # 全局排版参数
    plt.rcParams.update({'font.size': 22, 'font.family': 'serif'})
    
    file_paths = [
        "/sharenvme/usershome/cyl/test/prefill_data/重排实验：prefill_log_nocache_sort_maxsize=2_m.txt",
        "/sharenvme/usershome/cyl/test/prefill_data/重排实验：prefill_log_nocache_sort_re_maxsize=1_m.txt"
    ]
    
    # 【核心修复】：为两条曲线设置标准的学术名称
    label_map = {
        "重排实验：prefill_log_nocache_sort_maxsize=2_m.txt": "w/ Expert Reorder (Ours)",
        "重排实验：prefill_log_nocache_sort_re_maxsize=1_m.txt":"w/o Expert Reorder (Baseline)", 
    }
    
    custom_orange = (251/255, 125/255, 14/255)
    custom_blue = (0/255, 191/255, 255/255)
    
    # 颜色配置：第一条(无重排)设为蓝色，第二条(有重排)设为高亮橙色
    colors = [custom_blue, custom_orange]
    
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
            
            x_smooth, y_smooth = smooth_curve(lengths, avg_times)
            
            # 绘制平滑趋势线。将 Ours 的线条稍微加粗
            lw = 3.5 if "Ours" in label_name else 2.5
            z_order = 5 if "Ours" in label_name else 2
            ax.plot(x_smooth, y_smooth, linewidth=lw, label=label_name, color=colors[idx], alpha=0.9, zorder=z_order)
            
        # 坐标轴设置
        ax.set_xlabel('Prompt Length (# Tokens)', fontsize=24)
        ax.set_ylabel('Prefill Latency (s)', fontsize=26)
        
        # 锁定 Y 轴范围
        ax.set_ylim(10, 45)

        
        # 图例设置
        ax.legend(fontsize=18, frameon=False, loc='upper left')
        
        fig.tight_layout()
        plt.savefig('prefill_latency_smoothed_4.4.4_1.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('prefill_latency_smoothed_4.4.4_1.png', bbox_inches='tight')
        print("平滑绘图成功！已保存为 prefill_latency_smoothed_4.4.4_1.pdf")
    else:
        print("无法绘图：未读取到有效数据")