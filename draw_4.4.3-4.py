import matplotlib.pyplot as plt
import numpy as np

# 全局排版参数（学术风，衬线字体）
plt.rcParams.update({'font.size':19, 'font.family': 'serif'})

# 将中文策略映射为标准的学术英文术语，并按逻辑递进排序
labels = ['No Replace', 'Global Freq', 'Random', 'Ours']
# 将你提供的数据匹配到对应的标签上（0.263, 0.28, 0.31, 0.348）
hit_rates = np.array([0.263, 0.28, 0.31, 0.348]) * 100 

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 你的定制学术色
custom_blue = (0/255, 191/255, 255/255)
custom_orange = (251/255, 125/255, 14/255)

# 为 Ours 分配高亮橙色，其他 Baseline 分配标准蓝
colors = [custom_orange, custom_orange, custom_orange, custom_orange]

# 绘制柱状图，添加 edgecolor='black' 提升质感
bars = ax.bar(labels, hit_rates, width=0.4, color=colors, edgecolor='black', linewidth=1.2)

# 坐标轴设置
ax.set_ylabel('Expert Hit Rate (%)', fontsize=20)
ax.set_xlabel('Cache Replacement Policy', fontsize=20)



# 锁定 Y 轴范围到 45%，为顶部数据标签留出干净空间
ax.set_ylim(0, 45)


# 紧凑布局并保存
fig.tight_layout()
plt.savefig('cache_replacement_ablation_4.4.3.pdf', format='pdf', bbox_inches='tight')
plt.savefig('cache_replacement_ablation.png', bbox_inches='tight')
print("缓存替换策略柱状图已生成：cache_replacement_ablation.pdf")