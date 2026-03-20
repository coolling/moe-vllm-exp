# import matplotlib.pyplot as plt
# import numpy as np

# # ================= 全局排版参数 =================
# plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})

# # X轴标签与实验数据
# labels = ['Full', 'On-Demand', 'Seq', 'Static Cache', 'Ours']
# # 数据转换为秒 (s/token)
# mixtral_data = np.array([405.05, 4758.51, 4670.14, 3343.35, 2961.09]) / 1000.0
# qwen_data = np.array([97.82, 253.55, 248.55, 201.83, 184.00]) / 1000.0

# x = np.arange(len(labels))
# width = 0.35

# # 定制颜色：学术蓝与学术橙
# custom_blue = (0/255, 191/255, 255/255)
# custom_orange = (251/255, 125/255, 14/255)

# # ================= 创建带有断层 Y 轴的上下两个子图 =================
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), dpi=300, 
#                                gridspec_kw={'height_ratios': [1.8, 1]})

# # 缩小上下子图的物理间距，消除多余空白
# fig.subplots_adjust(hspace=0.03)

# # 在两个子图上都绘制并排的柱状图
# for ax in [ax1, ax2]:
#     ax.bar(x - width/2, mixtral_data, width=width, label='Mixtral-8x7B', 
#            color=custom_blue, edgecolor='black', linewidth=1.2)
#     ax.bar(x + width/2, qwen_data, width=width, label='Qwen1.5-MoE-A2.7B', 
#            color=custom_orange, edgecolor='black', linewidth=1.2)

# # ================= 设置各自的 Y 轴范围 =================
# # 顶部子图显示较大的延迟范围
# ax1.set_ylim(2.5, 5.2)
# # 底部子图显示较小的延迟范围
# ax2.set_ylim(0.0, 0.6)

# # ================= 绘制断层线条效果 =================
# ax1.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.tick_params(labeltop=False, bottom=False)
# ax2.xaxis.tick_bottom()

# # 绘制断层斜线 (//)
# d = .012  # 斜线的尺寸
# kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, linewidth=1.5)
# ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左上斜线
# ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右上斜线

# kwargs.update(transform=ax2.transAxes)        # 切换到底部子图坐标系
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左下斜线
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右下斜线

# # ================= 图例与标签设置 =================
# ax1.legend(loc='upper right', fontsize=16)
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)

# # 调整 Y 轴文字位置，使其更靠近坐标轴 (x=0.04)
# fig.supylabel('Decode Latency (s/token)', fontsize=20, x=0.04)

# # 保存文件
# plt.savefig('decode_latency_combined.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('decode_latency_combined.png', bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np

# ================= 全局排版参数 =================
plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})

# X轴标签与实验数据
labels = ['Full', 'On-Demand', 'Seq', 'Static Cache', 'Ours']
# 数据转换为秒 (s/token)
mixtral_data = np.array([405.05, 4758.51, 4670.14, 3343.35, 2961.09]) / 1000.0
qwen_data = np.array([97.82, 253.55, 248.55, 201.83, 184.00]) / 1000.0

x = np.arange(len(labels))
width = 0.35

# 定制颜色：学术蓝与学术橙
custom_blue = (0/255, 191/255, 255/255)
custom_orange = (251/255, 125/255, 14/255)

# ================= 创建带有断层 Y 轴的上下两个子图 =================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), dpi=300, 
                               gridspec_kw={'height_ratios': [1.8, 1]})

# 缩小上下子图的物理间距，消除多余空白
fig.subplots_adjust(hspace=0.06)

# 在两个子图上都绘制并排的柱状图
for ax in [ax1, ax2]:
    ax.bar(x - width/2, mixtral_data, width=width, label='Mixtral-8x7B', 
           color=custom_blue, edgecolor='black', linewidth=1.2)
    ax.bar(x + width/2, qwen_data, width=width, label='Qwen1.5-MoE-A2.7B', 
           color=custom_orange, edgecolor='black', linewidth=1.2)

# ================= 设置各自的 Y 轴范围 =================
# 顶部子图显示较大的延迟范围
ax1.set_ylim(2.5, 5.2)
# 底部子图显示较小的延迟范围
ax2.set_ylim(0.0, 0.8)

# ================= 绘制断层线条效果 =================
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.xaxis.tick_bottom()

# 绘制断层斜线 (//)
d = .012  # 斜线的尺寸
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, linewidth=1.5)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左上斜线
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右上斜线

kwargs.update(transform=ax2.transAxes)        # 切换到底部子图坐标系
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左下斜线
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右下斜线

# ================= 图例与标签设置 =================
ax1.legend(loc='upper right', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

# 调整 Y 轴文字位置，使其更靠近坐标轴 (x=0.04)
fig.supylabel('Decode Latency (s/token)', fontsize=20, x=0.04)

# 保存文件
plt.savefig('decode_latency_combined.pdf', format='pdf', bbox_inches='tight')
plt.savefig('decode_latency_combined.png', bbox_inches='tight')