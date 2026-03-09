import matplotlib.pyplot as plt
import numpy as np

# 数据准备：Layer 6 (给定 Layer 3,4,5 历史)
data_counts = {"0": 3991, "2": 2701, "5": 922, "3": 587, "4": 384, "1": 368, "7": 218, "6": 143}
total = sum(data_counts.values())
dist_percentages = [data_counts.get(str(i), 0) / total * 100 for i in range(8)]
expert_ids = np.arange(8)

# 使用你指定的橙色
custom_orange = (251/255, 125/255, 14/255)

# 学术排版参数（超大字体）
plt.rcParams.update({
    'font.size': 24, 'font.family': 'serif',
    'axes.labelsize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
})


fig, ax = plt.subplots(figsize=(8, 6))

# 绘制橙色折线，使用白底橙边的空心圆形标记
ax.plot(expert_ids, dist_percentages,
         color=custom_orange, linewidth=3,
         marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2)

ax.set_ylabel('Conditional Probability (%)')
ax.set_xlabel('Expert ID')
ax.set_xticks(expert_ids)

# 统一 Y 轴上限
ax.set_ylim(0, 55)
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig('4_2_4_trace1_line.pdf', format='pdf', bbox_inches='tight')
plt.savefig('4_2_4_trace1_line.png', bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np

# 数据准备：Layer 12 (给定 Layer 9,10,11 历史)
data_counts = {"3": 599, "5": 468, "2": 46, "6": 38, "0": 36, "1": 31, "7": 14, "4": 12}
total = sum(data_counts.values())
dist_percentages = [data_counts.get(str(i), 0) / total * 100 for i in range(8)]
expert_ids = np.arange(8)

# 使用你指定的蓝色
custom_blue = (0/255, 191/255, 255/255)

# 学术排版参数（超大字体）
plt.rcParams.update({
    'font.size': 24, 'font.family': 'serif',
    'axes.labelsize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
})

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制蓝色折线，使用白底蓝边的空心圆形标记
ax.plot(expert_ids, dist_percentages,
         color=custom_orange, linewidth=3,
         marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2)

ax.set_ylabel('Conditional Probability (%)')
ax.set_xlabel('Expert ID')
ax.set_xticks(expert_ids)

# 统一 Y 轴上限
ax.set_ylim(0, 55)
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig('4_2_4_trace2_line.pdf', format='pdf', bbox_inches='tight')
plt.savefig('4_2_4_trace2_line.png', bbox_inches='tight')