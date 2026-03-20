# import matplotlib.pyplot as plt
# import numpy as np

# # ================= 全局排版参数 =================
# plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})

# # 算法名称映射
# labels = ['Random', 'Sequential', 'Ours']

# # 定制颜色：基线为学术蓝，Ours 为学术橙
# custom_orange = (251/255, 125/255, 14/255)
# custom_blue = (0/255, 191/255, 255/255)
# colors = [custom_blue, custom_blue, custom_blue]

# # X 轴坐标
# x = np.arange(len(labels))
# width = 0.45  # 稍微加宽一点柱子，因为是单图

# # ================= 图 1: Mixtral-8x7B =================
# # 数据准备 (转换为秒 s/token)
# mixtral_data = np.array([3335.638, 3323.735, 2961.085]) / 1000.0

# fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=300)

# bars1 = ax1.bar(labels, mixtral_data, width=width, color=colors, edgecolor='black', linewidth=1.2)

# ax1.set_ylabel('Decode Latency (s/token)', fontsize=20)
# # 如果需要在图内显示标题，取消下面这行的注释即可
# # ax1.set_title('Mixtral-8x7B', fontsize=20, pad=15)


# # 锁定 Y 轴范围并移除边框
# ax1.set_ylim(2.2, 3.6)


# fig1.tight_layout()
# fig1.savefig('prediction_latency_mixtral.pdf', format='pdf', bbox_inches='tight')
# fig1.savefig('prediction_latency_mixtral.png', bbox_inches='tight')
# print("Mixtral 延迟单图已生成：prediction_latency_mixtral.pdf")


# # ================= 图 2: Qwen1.5-MoE-A2.7B =================
# # 数据准备 (保留毫秒 ms/token)
# qwen_data = np.array([195.0, 194.5, 184.0])

# fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
# colors = [custom_orange, custom_orange, custom_orange]
# bars2 = ax2.bar(labels, qwen_data, width=width, color=colors, edgecolor='black', linewidth=1.2)

# ax2.set_ylabel('Decode Latency (ms/token)', fontsize=20)
# # 如果需要在图内显示标题，取消下面这行的注释即可
# # ax2.set_title('Qwen1.5-MoE-A2.7B', fontsize=20, pad=15)


# # 锁定 Y 轴范围并移除边框
# ax2.set_ylim(120, 220)


# fig2.tight_layout()
# fig2.savefig('prediction_latency_qwen.pdf', format='pdf', bbox_inches='tight')
# fig2.savefig('prediction_latency_qwen.png', bbox_inches='tight')
# print("Qwen 延迟单图已生成：prediction_latency_qwen.pdf")
import matplotlib.pyplot as plt
import numpy as np

# ================= 全局排版参数 =================
plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})

labels = ['Random', 'Sequential', 'Ours']
custom_blue = (0/255, 191/255, 255/255)
custom_orange = (251/255, 125/255, 14/255)

x = np.arange(len(labels))
width = 0.32

mixtral_data = np.array([3335.638, 3323.735, 2961.085]) / 1000.0
qwen_data = np.array([195.0, 194.5, 184.0]) / 1000.0

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), dpi=300, 
                               gridspec_kw={'height_ratios': [1.5, 1]})
# 修改：进一步缩小中间的空白间距，设为 0.02
fig.subplots_adjust(hspace=0.06)

for ax in [ax1, ax2]:
    ax.bar(x - width/2, mixtral_data, width=width, label='Mixtral-8x7B', 
           color=custom_blue, edgecolor='black', linewidth=1.2)
    ax.bar(x + width/2, qwen_data, width=width, label='Qwen1.5-MoE-A2.7B', 
           color=custom_orange, edgecolor='black', linewidth=1.2)

ax1.set_ylim(2.8, 3.6)
ax2.set_ylim(0.0, 0.25)

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.xaxis.tick_bottom()

d = .015  
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, linewidth=1.5)
ax1.plot((-d, +d), (-d, +d), **kwargs)        
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  

kwargs.update(transform=ax2.transAxes)        
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

ax1.legend(loc='upper right', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

# 修改：调整 x 的值，让标签离坐标轴更近（原来是 0.02，现在改成 0.06 试试）
fig.supylabel('Decode Latency (s/token)', fontsize=20, x=0.02)

fig.savefig('test.png', bbox_inches='tight')
fig.savefig('prediction_latency.pdf', format='pdf', bbox_inches='tight')