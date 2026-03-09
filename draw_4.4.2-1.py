import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（大字体，适合论文，使用 serif 衬线字体）
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# 调整 figsize 以适应 5 组双柱的显示
fig, ax = plt.subplots(figsize=(10, 5.5))

labels = ['Full Resident', 'On-Demand', 'Seq Prefetch', 'Static Cache', 'Ours']
mixtral_data = [89.0, 4.8, 6.88, 26.48, 26.48]
qwen_data = [28.7, 5.6, 6.57, 11.77, 11.77]

x = np.arange(len(labels))
width = 0.35

# 你要求的定制颜色
custom_orange = (251/255, 125/255, 14/255)
custom_blue = (0/255, 191/255, 255/255)

# 绘制柱状图，加上 edgecolor='black' 是提升学术图表质感的关键
rects1 = ax.bar(x - width/2, mixtral_data, width, label='Mixtral-8x7B', color=custom_blue, edgecolor='black')
rects2 = ax.bar(x + width/2, qwen_data, width, label='Qwen1.5-MoE-A2.7B', color=custom_orange, edgecolor='black')

# 坐标轴与标签设置
ax.set_ylabel('Memory Footprint (GB)', fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=18)
ax.legend(fontsize=20)

# 在柱子上标注具体数值（保留数值标注以便评委直观看出对比）
# ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=12)
# ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=12)

# 按照要求，去掉了所有的网格虚线 (grid)
# ax.grid(axis='y', linestyle='--', alpha=0.6) 

fig.tight_layout()
plt.savefig('memory_footprint_styled.pdf', format='pdf', bbox_inches='tight')
plt.savefig('memory_footprint_styled.png', bbox_inches='tight')