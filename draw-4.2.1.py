import matplotlib.pyplot as plt
import numpy as np
custom_orange = (251/255, 125/255, 14/255)
custom_blue = (0/255, 191/255, 255/255)
# 数据准备
models = ['Mixtral', 'Qwen1.5-MoE']
attn_compute = [60.8, 21.6]
moe_compute = [344.2, 76.21]

x = np.arange(len(models))
width = 0.35

# 设置全局字体大小，适应学术论文
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# 绘制柱状图
ax.bar(x - width/2, attn_compute, width, label='Attn', color=custom_orange, edgecolor='black')
ax.bar(x + width/2, moe_compute, width, label='MoE', color=custom_blue, edgecolor='black')

ax.set_ylabel('Execution Time (ms)') # 假设时间单位为 ms，请根据实际情况修改
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig('4_2_1_execution_time.pdf', format='pdf', bbox_inches='tight')
plt.savefig('4_2_1_execution_time.png', bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['Mixtral', 'Qwen1.5-MoE']
attn_memory = [3.0, 3.46]
moe_memory = [84.0, 23.2]

x = np.arange(len(models))
width = 0.35

# 设置全局字体大小，适应学术论文
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# 绘制柱状图
ax.bar(x - width/2, attn_memory, width, label='Attn', color=custom_orange, edgecolor='black')
ax.bar(x + width/2, moe_memory, width, label='MoE', color=custom_blue, edgecolor='black')

ax.set_ylabel('Memory Footprint (GB)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig('4_2_1_memory_footprint.pdf', format='pdf', bbox_inches='tight')
plt.savefig('4_2_1_memory_footprint.png',bbox_inches='tight')
plt.close()