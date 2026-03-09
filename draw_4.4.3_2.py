import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# 实验数据
cache_sizes = np.array([0, 1, 2, 3, 4])
memory_footprint = np.array([7.175, 16.675, 26.475, 36.275, 46.175])

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 定制学术蓝
custom_blue = (0/255, 191/255, 255/255)

# 绘制柱状图，添加 edgecolor='black' 提升学术图表质感
bars = ax.bar(cache_sizes, memory_footprint, width=0.5, color=custom_blue, edgecolor='black', linewidth=1.2)

# 坐标轴设置
ax.set_xlabel('Number of Cached Experts per Layer', fontsize=20)
ax.set_ylabel('Memory Footprint (GB)', fontsize=20)
ax.set_xticks(cache_sizes)


# 锁定 Y 轴范围，为顶部数据标签留出空间
ax.set_ylim(0, 55)


# 紧凑布局并保存高分辨率图片与 PDF 矢量图
fig.tight_layout()
plt.savefig('memory_footprint_ablation_4.4.3.pdf', format='pdf', bbox_inches='tight')
plt.savefig('memory_footprint_ablation.png', bbox_inches='tight')
print("内存占用消融图表已生成：memory_footprint_ablation.pdf")