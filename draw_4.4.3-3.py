import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# 实验数据
cache_sizes = np.array([1, 2, 3, 4])
# 将原始小数转换为百分比 (%) 用于绘图
hit_rates = np.array([0.191, 0.348, 0.4824, 0.608]) * 100 

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 定制学术蓝
custom_orange = (251/255, 125/255, 14/255)

# 绘制柱状图，添加 edgecolor='black' 提升学术图表质感
bars = ax.bar(cache_sizes, hit_rates, width=0.4, color=custom_orange, edgecolor='black', linewidth=1.2)

# 坐标轴设置
ax.set_xlabel('Number of Cached Experts per Layer', fontsize=20)
ax.set_ylabel('Expert Hit Rate (%)', fontsize=20)
ax.set_xticks(cache_sizes)



# 锁定 Y 轴范围到 75%，为顶部数据标签留出干净的呼吸空间
ax.set_ylim(0, 75)



# 紧凑布局并保存高分辨率图片与 PDF 矢量图
fig.tight_layout()
plt.savefig('expert_hit_rate_ablation_4.4.3.pdf', format='pdf', bbox_inches='tight')
plt.savefig('expert_hit_rate_ablation.png', bbox_inches='tight')
print("专家命中率柱状图已生成：expert_hit_rate_ablation.pdf")