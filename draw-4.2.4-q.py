import matplotlib.pyplot as plt
import numpy as np

# 1. 全局学术排版参数设置
plt.rcParams.update({
    'font.size': 28, 'font.family': 'serif',
    'axes.labelsize': 28, 'xtick.labelsize': 28, 'ytick.labelsize': 28,
})

# 定义颜色
custom_orange = (251/255, 125/255, 14/255)
custom_blue = (0/255, 191/255, 255/255)

# ==========================================
# 图 1：Layer 6 (给定 Layer 3,4,5 历史)
# ==========================================
data_counts_1 = {
  "0": 206, "1": 2, "2": 3, "3": 5, "4": 2, "5": 230, "6": 20, "7": 91, "8": 6, "9": 99, 
  "10": 5, "11": 11, "12": 6, "13": 63, "14": 84, "15": 65, "16": 11, "17": 5, "18": 23, "19": 2, 
  "20": 1, "21": 11, "22": 2, "23": 6, "24": 1, "25": 33, "26": 6, "27": 7, "28": 1, "29": 10, 
  "30": 4, "31": 92, "32": 25, "33": 3, "34": 11, "35": 19, "36": 1, "37": 3, "38": 7, "39": 1, 
  "40": 10, "41": 145, "42": 10, "43": 10, "44": 14, "45": 10, "46": 1, "47": 9, "48": 15, "49": 1, 
  "50": 466, "51": 1, "52": 2, "53": 1, "54": 5, "55": 8, "56": 28, "57": 25, "58": 6, "59": 1
}

total_1 = sum(data_counts_1.values())
dist_percentages_1 = [data_counts_1.get(str(i), 0) / total_1 * 100 for i in range(60)]
expert_ids_1 = np.arange(60)

fig1, ax1 = plt.subplots(figsize=(8, 6))

# 绘制橙色折线，调整线宽和点大小以适应 60 个点
ax1.plot(expert_ids_1, dist_percentages_1,
         color=custom_orange, linewidth=2,
         marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.5)
ax1.set_ylabel('Conditional Probability (%)')
ax1.set_xlabel('Expert ID')
ax1.set_xticks(np.arange(0, 61, 10)) # 每隔 10 显示一个刻度
ax1.set_ylim(0, 30)

fig1.tight_layout()
fig1.savefig('4_2_4_trace1_line_q.pdf', format='pdf', bbox_inches='tight')
fig1.savefig('4_2_4_trace1_line_q.png', bbox_inches='tight')


# ==========================================
# 图 2：Layer 12 (给定 Layer 9,10,11 历史)
# ==========================================
data_counts_2 = {
  "0": 8, "1": 5, "2": 1, "3": 14, "4": 13, "5": 19, "6": 5, "7": 3, "8": 1, "9": 1, 
  "10": 1, "11": 2, "12": 1, "13": 15, "14": 1, "15": 19, "16": 15, "17": 35, "18": 3, "19": 2, 
  "20": 21, "21": 6, "22": 4, "23": 4, "24": 20, "25": 73, "26": 9, "27": 3, "28": 16, "29": 47, 
  "30": 18, "31": 71, "32": 1, "33": 10, "34": 31, "35": 41, "36": 111, "37": 1, "38": 3, "39": 30, 
  "40": 15, "41": 1, "42": 8, "43": 1, "44": 1, "45": 2, "46": 1, "47": 1, "48": 38, "49": 25, 
  "50": 19, "51": 1, "52": 1, "53": 17, "54": 1, "55": 6, "56": 15, "57": 16, "58": 63, "59": 11
}

total_2 = sum(data_counts_2.values())
dist_percentages_2 = [data_counts_2.get(str(i), 0) / total_2 * 100 for i in range(60)]
expert_ids_2 = np.arange(60)

fig2, ax2 = plt.subplots(figsize=(8, 6))

# 绘制蓝色折线
ax2.plot(expert_ids_2, dist_percentages_2,
         color=custom_blue, linewidth=2,
         marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.5)

ax2.set_ylabel('Conditional Probability (%)')
ax2.set_xlabel('Expert ID')
ax2.set_xticks(np.arange(0, 61, 10)) # 每隔 10 显示一个刻度
ax2.set_ylim(0, 15)

fig2.tight_layout()
fig2.savefig('4_2_4_trace2_line_q.pdf', format='pdf', bbox_inches='tight')
fig2.savefig('4_2_4_trace2_line_q.png', bbox_inches='tight')

# 显式关闭图形释放内存
plt.close('all')