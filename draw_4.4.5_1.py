import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 16, 'font.family': 'serif'})

# 算法名称映射为标准学术英文，并按逻辑递进排序
labels = ['Random', 'Sequential', 'Request Freq', 'Path-based\n(Ours)']

# 原始数据提取并按 [Random, Sequential, Request Freq, Path-based(Ours)] 重新排列
# Mixtral: 随机(0.23), 顺序(0.25), 请求频率(0.36), 路径相似(0.60)
mixtral_data = np.array([0.23, 0.25, 0.36, 0.60]) * 100
# Qwen: 随机(0.065), 顺序(0.064), 请求频率(0.165), 路径相似(0.245)
qwen_data = np.array([0.065, 0.064, 0.165, 0.245]) * 100

x = np.arange(len(labels))
width = 0.35  # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 定制颜色：为两个模型分配不同颜色
color_mixtral = '#1f77b4'  # 学术蓝
color_qwen = '#ff7f0e'     # 学术橙

# 绘制分组柱状图
rects1 = ax.bar(x - width/2, mixtral_data, width, label='Mixtral-8x7B', color=color_mixtral, edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x + width/2, qwen_data, width, label='Qwen1.5-MoE-A2.7B', color=color_qwen, edgecolor='black', linewidth=1.2)

# 坐标轴与标签设置
ax.set_ylabel('Prediction Accuracy (%)', fontsize=18)
ax.set_xlabel('Expert Prediction Algorithm', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)

# 添加图例
ax.legend(fontsize=14, frameon=False, loc='upper left')

# 在柱子上直接标注百分比数值（保留一位小数，粗体）
ax.bar_label(rects1, padding=3, fmt='%.1f%%', fontsize=12, fontweight='bold')
ax.bar_label(rects2, padding=3, fmt='%.1f%%', fontsize=12, fontweight='bold')

# 锁定 Y 轴范围，为顶部数据标签和图例留出空间
ax.set_ylim(0, 80)

# 移除顶部和右侧的边框线 (顶会标准)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 紧凑布局并保存
fig.tight_layout()
plt.savefig('prediction_accuracy_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('prediction_accuracy_comparison.png', bbox_inches='tight')
print("预测准确率柱状图已生成：prediction_accuracy_comparison.pdf")