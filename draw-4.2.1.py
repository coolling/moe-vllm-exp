import matplotlib.pyplot as plt
import numpy as np

# 颜色设置
# custom_orange = (251/255, 125/255, 14/255) # 原来的橙色
custom_green = (0/255, 147/255, 0/255)       # 新绿色 (009300)
custom_blue = (0/255, 191/255, 255/255)

# 设置全局字体和大小，适应学术论文，指定 Times New Roman 字体
plt.rcParams.update({
    'font.size': 18,               # 字号，可根据实际排版微调
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

# ==================== 绘制图 1：Execution Time ====================
models = ['Mixtral', 'Qwen1.5-MoE']
attn_compute = [60.8, 21.6]
moe_compute = [344.2, 76.21]

x = np.arange(len(models))
width = 0.3  # 稍微加宽一点让图更饱满

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# 使用新绿色，加上 zorder=3 使柱状图位于网格线上方，设置 linewidth 加粗边线
ax.bar(x - width/2, attn_compute, width, label='Attn', color=custom_green, edgecolor='black', linewidth=1.5, zorder=3)
ax.bar(x + width/2, moe_compute, width, label='MoE', color=custom_blue, edgecolor='black', linewidth=1.5, zorder=3)

ax.set_ylabel('Execution Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(models)

# 确保所有边框都加上 (Spines)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# 添加水平网格线，透明度调低作为辅助阅读，使用实线 (linestyle='-')
ax.grid(axis='y', linestyle='-', alpha=0.4, zorder=0)

# 图例加上边框 (frameon=True)
ax.legend(frameon=True)

fig.tight_layout()
plt.savefig('4_2_1_execution_time.pdf', format='pdf', bbox_inches='tight')
plt.savefig('4_2_1_execution_time.png', bbox_inches='tight', dpi=300)
plt.close()


# ==================== 绘制图 2：Memory Footprint ====================
attn_memory = [3.0, 3.46]
moe_memory = [84.0, 23.2]

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# 同样保持样式统一，使用新绿色
ax.bar(x - width/2, attn_memory, width, label='Attn', color=custom_green, edgecolor='black', linewidth=1.5, zorder=3)
ax.bar(x + width/2, moe_memory, width, label='MoE', color=custom_blue, edgecolor='black', linewidth=1.5, zorder=3)

ax.set_ylabel('Memory Footprint (GB)')
ax.set_xticks(x)
ax.set_xticklabels(models)

# 确保所有边框都加上 (Spines)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# 开启 y 轴辅助线，使用实线
ax.grid(axis='y', linestyle='-', alpha=0.4, zorder=0)

# 图例加上边框
ax.legend(frameon=True)

fig.tight_layout()
plt.savefig('4_2_1_memory_footprint.pdf', format='pdf', bbox_inches='tight')
plt.savefig('4_2_1_memory_footprint.png', bbox_inches='tight', dpi=300)
plt.close()