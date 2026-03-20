import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})

# 算法名称映射为标准学术英文，并按逻辑递进排序
labels = ['Random', 'Sequential', 'Freq', 'Ours']

# 原始数据提取并按 [Random, Sequential, Request Freq, Path-based(Ours)] 重新排列
# Mixtral: 随机(0.23), 顺序(0.25), 请求频率(0.36), 路径相似(0.60)
mixtral_data = np.array([0.23, 0.25, 0.36, 0.60]) * 100
# Qwen: 随机(0.065), 顺序(0.064), 请求频率(0.165), 路径相似(0.245)
qwen_data = np.array([0.065, 0.064, 0.165, 0.245]) * 100

x = np.arange(len(labels))
width = 0.32  # 柱子宽度

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 定制颜色：为两个模型分配不同颜色
custom_orange = (251/255, 125/255, 14/255)
custom_blue = (0/255, 191/255, 255/255)
color_mixtral = custom_blue # 学术蓝
color_qwen = custom_orange    # 学术橙

# 绘制分组柱状图
rects1 = ax.bar(x - width/2, mixtral_data, width, label='Mixtral-8x7B', color=color_mixtral, edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x + width/2, qwen_data, width, label='Qwen1.5-MoE-A2.7B', color=color_qwen, edgecolor='black', linewidth=1.2)

# 坐标轴与标签设置
ax.set_ylabel('Prediction Accuracy (%)', fontsize=20)
# ax.set_xlabel('Expert Prediction Algorithm', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)

# 添加图例
ax.legend(fontsize=18, frameon=False, loc='upper left')


# 锁定 Y 轴范围，为顶部数据标签和图例留出空间
ax.set_ylim(0, 80)


# 紧凑布局并保存
fig.tight_layout()
plt.savefig('prediction_accuracy_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('prediction_accuracy_comparison.png', bbox_inches='tight')
print("预测准确率柱状图已生成：prediction_accuracy_comparison.pdf")