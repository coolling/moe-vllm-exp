import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# 实验数据
cache_sizes = np.array([0, 1, 2, 3, 4])
# 【修改点 1】：将毫秒数据除以 1000 转换为秒
latencies = np.array([4400.527, 3566.745, 2961.085, 2442.209, 2027.149]) / 1000.0

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 你的定制学术蓝
custom_blue = (0/255, 191/255, 255/255)

# 绘制带有方块 Marker 的折线图，加粗线条和边框以提升 PDF 印刷质感
ax.plot(cache_sizes, latencies, marker='s', markersize=10, 
        linewidth=3, color=custom_blue, markeredgecolor='black', markeredgewidth=1.5)

# 坐标轴设置
ax.set_xlabel('Number of Cached Experts per Layer', fontsize=20)
# 【修改点 2】：将标签修改为 s/token
ax.set_ylabel('Decode Latency (s/token)', fontsize=20)
ax.set_xticks(cache_sizes)

# 【修改点 3】：标注数值保留2位小数，并将上移的偏移量从 100(ms) 改为 0.15(s)
# for i, (x, y) in enumerate(zip(cache_sizes, latencies)):
#     ax.text(x, y + 0.15, f'{y:.2f}', ha='center', va='bottom', fontsize=20, fontweight='bold')

# 【修改点 4】：将 Y 轴范围按比例缩小到 1.5 ~ 5.5 秒，给顶部标签留出呼吸空间
ax.set_ylim(1.5, 5.5)


# 紧凑布局并保存
fig.tight_layout()
plt.savefig('cache_size_ablation_4.4.3.pdf', format='pdf', bbox_inches='tight')
plt.savefig('cache_size_ablation.png', bbox_inches='tight')
print("缓存消融图表（秒级单位）已生成：cache_size_ablation_4.4.3.pdf")