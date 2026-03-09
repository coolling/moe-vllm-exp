import matplotlib.pyplot as plt
import numpy as np

# 实测数据
expert_ids = np.arange(8)
token_counts = np.array([14, 7, 58, 13, 18, 74, 68, 18])

# 全局排版参数（大字体，适合论文）
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(7, 4.5))

# 根据负载轻重分配颜色，凸显对比
# 大于 50 个 Token 的视为重负载 (专家 2, 5, 6)，其他为轻负载
custom_orange = (251/255, 125/255, 14/255)
custom_blue = (0/255, 191/255, 255/255)
colors = [custom_orange]
edges = ['#D98C46']

bars = ax.bar(expert_ids, token_counts, color=colors, edgecolor='black', width=0.6)

# 坐标轴设置
ax.set_xlabel('Expert ID', fontsize=16)
ax.set_ylabel('Assigned Tokens (Load)', fontsize=16)
ax.set_xticks(expert_ids)

# # 在柱子上标注具体数值（让评委直观看到 7 和 74 的巨大落差）
# for bar in bars:
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), 
#             ha='center', va='bottom', fontsize=12)



ax.grid(axis='y', linestyle='--', alpha=0.6)

fig.tight_layout()
plt.savefig('4_2_3_prefill_load_imbalance.pdf', format='pdf', bbox_inches='tight')
plt.savefig('4_2_3_prefill_load_imbalance.png', bbox_inches='tight')