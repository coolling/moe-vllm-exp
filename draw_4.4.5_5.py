import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})

# 实验数据：提前预测层数 (Lookahead Distance k)
lookahead_layers = np.array([1, 2, 3, 4, 5])
# 预测表的内存占用数据 (MB)
memory_footprint = np.array([0.27, 4.07, 69.56, 766.0, 2532.0])

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 定制学术橙（对应 Ours 机制的内部消融）
custom_blue = (0/255, 191/255, 255/255)

# 绘制柱状图，添加 edgecolor='black' 提升学术质感
bars = ax.bar(lookahead_layers, memory_footprint, width=0.5, color=custom_blue, edgecolor='black', linewidth=1.2)

# 坐标轴设置
ax.set_xlabel('Lookahead Layers', fontsize=20)
# 系统术语：元数据内存占用（Metadata Memory Footprint），因为这不是模型权重，是预测算法本身的开销
ax.set_ylabel('Memory Footprint of CPT (MB)', fontsize=20)
ax.set_xticks(lookahead_layers)

# 为避免数值重叠，前三个保留两位小数，后两个庞大数值直接取整
labels = [f'{val:.2f}' if val < 100 else f'{int(val)}' for val in memory_footprint]
ax.bar_label(bars, labels=labels, padding=5, fontsize=14, fontweight='bold')

# 锁定 Y 轴范围。为顶部标签留出呼吸空间
ax.set_ylim(0, 3000)



# 紧凑布局并保存
fig.tight_layout()
plt.savefig('prefetch_lookahead_memory_bar.pdf', format='pdf', bbox_inches='tight')
plt.savefig('prefetch_lookahead_memory_bar.png', bbox_inches='tight')
print("预取层数-内存占用柱状图已生成：prefetch_lookahead_memory_bar.pdf")