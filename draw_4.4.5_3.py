import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# 实验数据：提前预测层数 (Lookahead Distance k)
lookahead_layers = np.array([1, 2, 3, 4, 5])
# 将 ms/token 转换为 s/token
latencies = np.array([3129.966, 3025.417, 2961.085, 2910.373, 2890.304]) / 1000.0

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 定制学术橙（因为这是对 Ours 策略的深入剖析）
custom_orange = (251/255, 125/255, 14/255)

# 绘制带有圆形 Marker 的折线图，加粗线条和边框以提升 PDF 印刷质感
ax.plot(lookahead_layers, latencies, marker='o', markersize=10, 
        linewidth=3, color=custom_orange, markeredgecolor='black', markeredgewidth=1.5)

# 坐标轴设置
ax.set_xlabel('Lookahead Layers', fontsize=22)
ax.set_ylabel('Decode Latency (s/token)', fontsize=22)
ax.set_xticks(lookahead_layers)



# 锁定 Y 轴范围，为顶部数据标签留出呼吸空间
ax.set_ylim(2.7, 3.25)



# 紧凑布局并保存高分辨率图片与 PDF 矢量图
fig.tight_layout()
plt.savefig('prefetch_lookahead_ablation.pdf', format='pdf', bbox_inches='tight')
plt.savefig('prefetch_lookahead_ablation.png', bbox_inches='tight')
print("预取层数消融图表已生成：prefetch_lookahead_ablation.pdf")