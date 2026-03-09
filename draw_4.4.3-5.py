import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 19, 'font.family': 'serif'})

# 将中文策略映射为标准的学术英文术语，并按表现从差到好排序
labels = ['No Replace', 'Global Freq', 'Random', 'Ours']

# 填入你提供的数据（单位：ms），并除以 1000 转换为秒 (s/token)
latencies = np.array([3215.683, 3150.0, 3070.0, 2961.085]) / 1000.0

# ================= 核心修改：创建上下两个子图来实现断轴 =================
# gridspec_kw 控制上下两部分的比例，这里设置上半部分占 5 份，下半部分占 1 份
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), dpi=300, 
                               gridspec_kw={'height_ratios': [5, 1]})
fig.subplots_adjust(hspace=0.08)  # 控制上下断层的间距

# 定制学术色
custom_blue = (0/255, 191/255, 255/255)
# 提示：如果你想让 Ours 更显眼，可以把最后一个颜色换成 custom_orange
colors = [custom_blue, custom_blue, custom_blue, custom_blue] 

# 在上下两个子图都画出一样的柱状图
bars1 = ax1.bar(labels, latencies, width=0.4, color=colors, edgecolor='black', linewidth=1.2)
bars2 = ax2.bar(labels, latencies, width=0.4, color=colors, edgecolor='black', linewidth=1.2)

# ================= 设置 Y 轴截断范围 =================
# 上半部分显示 2.5 到 3.3（放大数据差异）
ax1.set_ylim(2.85, 3.3)
# 下半部分只显示 0 到 0.2（保留原点）
ax2.set_ylim(0, 0.2)

# ================= 隐藏边框，制造断轴错觉 =================
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)


# 隐藏上半部分的 X 轴刻度，避免重复
ax1.tick_params(bottom=False)

# ================= 绘制断轴的 "//" 斜线符号 =================
d = 0.015  # 斜线的大小
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, linewidth=1.5)
ax1.plot((-d, +d), (-d, +d), **kwargs)          # 左上截断线
# ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 如果右侧边框保留，就需要画右上截断线（这里右边框隐藏了所以不画）

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)    # 左下截断线

# ================= 坐标轴标签与数值标注 =================
# 将 Y 轴标签统一设置在中间偏左的位置
fig.text(0.02, 0.5, 'Decode Latency (s/token)', va='center', rotation='vertical', fontsize=20)
ax2.set_xlabel('Cache Replacement Policy', fontsize=20)

# 仅在上半部分的柱子上标注具体数值（保留两位小数）
# ax1.bar_label(bars1, padding=5, fmt='%.2f', fontsize=16, fontweight='bold')

# 保存高清矢量图
plt.savefig('cache_replacement_latency_4.4.3.pdf', format='pdf', bbox_inches='tight')
plt.savefig('cache_replacement_latency.png', bbox_inches='tight')
print("断轴柱状图已生成：cache_replacement_latency.pdf")