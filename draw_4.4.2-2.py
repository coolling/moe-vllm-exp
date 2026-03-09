import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（大字体，适合论文，使用 serif 衬线字体）
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# 创建图表，调整 figsize 以适应 5 组双柱的显示
fig, ax = plt.subplots(figsize=(10, 5.5))

# X轴标签与实验数据
labels = ['Full Resident', 'On-Demand', 'Seq Prefetch', 'Static Cache', 'Ours']
mixtral_data = [405.05, 4758.51, 4670.14, 3343.35, 2961.09]
qwen_data = [97.82, 253.55, 248.55, 201.83, 184.00]

x = np.arange(len(labels))
width = 0.35

# 严格采用你要求的定制颜色
custom_orange = (251/255, 125/255, 14/255)
custom_blue = (0/255, 191/255, 255/255)

# 绘制柱状图，添加 edgecolor='black' 提升学术图表质感
rects1 = ax.bar(x - width/2, mixtral_data, width, label='Mixtral-8x7B', color=custom_blue, edgecolor='black')
rects2 = ax.bar(x + width/2, qwen_data, width, label='Qwen1.5-MoE-A2.7B', color=custom_orange, edgecolor='black')

# 坐标轴与标签设置
ax.set_ylabel('Decode Latency (ms/token)', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.legend(fontsize=18)



# 紧凑布局并保存高分辨率图片与 PDF 矢量图
fig.tight_layout()
plt.savefig('decode_latency_4.4.2_2.pdf', format='pdf', bbox_inches='tight')
plt.savefig('decode_latency.png', bbox_inches='tight')