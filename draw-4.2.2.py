import matplotlib.pyplot as plt
import numpy as np

# 数据集 1：多请求分布（均匀）
data_multi = [
    {"expert": 7, "percentage": 15.5759}, {"expert": 5, "percentage": 13.6393},
    {"expert": 1, "percentage": 13.0762}, {"expert": 0, "percentage": 12.0981},
    {"expert": 4, "percentage": 12.084}, {"expert": 6, "percentage": 11.6447},
    {"expert": 2, "percentage": 11.3924}, {"expert": 3, "percentage": 10.4895}
]

# 数据集 2：单请求观察 A（专家 4, 7 为热点）
data_single_1 = [
    {"expert": 4, "percentage": 30.8358}, {"expert": 7, "percentage": 24.6469},
    {"expert": 1, "percentage": 12.7618}, {"expert": 2, "percentage": 9.9825},
    {"expert": 0, "percentage": 7.8422}, {"expert": 6, "percentage": 4.8222},
    {"expert": 3, "percentage": 4.7735}, {"expert": 5, "percentage": 4.3351}
]

# 数据集 3：单请求观察 B（专家 2, 3 为热点）
data_single_2 = [
    {"expert": 2, "percentage": 25.8907}, {"expert": 3, "percentage": 21.5768},
    {"expert": 5, "percentage": 10.7621}, {"expert": 7, "percentage": 11.6183},
    {"expert": 1, "percentage": 10.9267}, {"expert": 6, "percentage": 9.3361},
    {"expert": 0, "percentage": 5.6017}, {"expert": 4, "percentage": 4.2877}
]

# 数据提取与排序函数（确保 X 轴始终是专家 0~7）
def get_sorted_xy(data):
    sorted_data = sorted(data, key=lambda d: d['expert'])
    x = [str(d['expert']) for d in sorted_data]
    y = [d['percentage'] for d in sorted_data]
    return x, y

x1, y1 = get_sorted_xy(data_multi)
x2, y2 = get_sorted_xy(data_single_1)
x3, y3 = get_sorted_xy(data_single_2)

# 全局排版参数（大字体，适合论文）
plt.rcParams.update({'font.size': 26, 'font.family': 'serif'})

def plot_bar(x, y, title, filename, color):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(x, y, color=color, edgecolor='black', width=0.6)
    
    ax.set_ylabel('Activation Ratio(%)', fontsize=26)
    ax.set_xlabel('Expert ID', fontsize=29)
    
    # 核心学术技巧：固定 Y 轴上限，凸显局部倾斜的视觉落差
    # ax.set_ylim(0, 35)
    
    # ax.set_title(title, fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
def plot_bar_png(x, y, title, filename, color):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(x, y, color=color, edgecolor='black', width=0.6)
    
    ax.set_ylabel('Activation Ratio(%)', fontsize=26)
    ax.set_xlabel('Expert ID', fontsize=29)
    
    # 核心学术技巧：固定 Y 轴上限，凸显局部倾斜的视觉落差
    # ax.set_ylim(0, 35)
    
    # ax.set_title(title, fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(filename,  bbox_inches='tight')
    plt.close()
# 生成三张图 (使用不同的莫兰迪色系区分场景)
custom_orange = (251/255, 125/255, 14/255)
custom_blue = (0/255, 191/255, 255/255)
plot_bar(x1, y1, '(a) Multi-Request Activation', '4_2_2_multi_request.pdf', custom_orange)
plot_bar(x2, y2, '(b) Single-Request (Obs 1)', '4_2_2_single_1.pdf', custom_blue )
plot_bar(x3, y3, '(c) Single-Request (Obs 2)', '4_2_2_single_2.pdf', custom_blue)
plot_bar_png(x1, y1, '(a) Multi-Request Activation', '4_2_2_multi_request.png', custom_orange)
plot_bar_png(x2, y2, '(b) Single-Request (Obs 1)', '4_2_2_single_1.png', custom_blue )
plot_bar_png(x3, y3, '(c) Single-Request (Obs 2)', '4_2_2_single_2.png', custom_blue )
print("图表已生成！")