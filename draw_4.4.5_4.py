import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})

# 实验数据：提前预测层数 (Lookahead Distance k)
lookahead_layers = np.array([1, 2, 3, 4, 5])
# 将准确率小数转换为百分比 (%) 
accuracies = np.array([0.46, 0.54, 0.60, 0.62, 0.63]) * 100

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 定制学术橙
custom_orange = (251/255, 125/255, 14/255)

# 绘制柱状图，添加 edgecolor='black' 提升学术质感
bars = ax.bar(lookahead_layers, accuracies, width=0.5, color=custom_orange, edgecolor='black', linewidth=1.2)

# 坐标轴设置
ax.set_xlabel('Lookahead Layers', fontsize=20)
ax.set_ylabel('Prediction Accuracy (%)', fontsize=20)
ax.set_xticks(lookahead_layers)



# 锁定 Y 轴范围。注意：柱状图从 0 开始以保持数学严谨性，上限设为 75
ax.set_ylim(30, 70)



# 紧凑布局并保存高分辨率图片与 PDF 矢量图
fig.tight_layout()
plt.savefig('prefetch_lookahead_accuracy_bar.pdf', format='pdf', bbox_inches='tight')
plt.savefig('prefetch_lookahead_accuracy_bar.png', bbox_inches='tight')
print("预取层数-预测准确率柱状图已生成：prefetch_lookahead_accuracy_bar.pdf")