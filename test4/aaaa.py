import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime


#   values = np.sqrt(values)
#   values = np.sqrt(values)
#   values = np.sqrt(values)

# 读取数据
data = pd.read_csv('data.csv')

# 选择需要的属性
# selected_features = ['L', 'R', 'F', 'M', 'C', 'A', 'B', 'S1', 'Lfd']
# data = data[selected_features]
data.set_index('MEMBER_NO', inplace=True)

# 数据预处理
data = data.dropna()  # 删除缺失值
data["Lfd"] = data["Lfd"].replace("2014/2/29  0:00:00", "2014/2/28")
data['Lfd'] = pd.to_datetime(data['Lfd'], format='%Y/%m/%d')  # 时间转换
now = datetime.now()
data['Lfd'] = (now - data['Lfd']) / np.timedelta64(1, 'M')  # 计算时间差，单位为月
data['Lfd'] = data['Lfd'].astype(int)
data_scaled = StandardScaler().fit_transform(data)  # 数据标准化

# 寻找最优的K值
# silhouette_scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#    # 使用轮廓系数作为评价指标
#     score = silhouette_score(data_scaled, kmeans.fit_predict(data_scaled))
#     silhouette_scores.append(score)
#     print(f"Success for k = {k}, score = {score:.3f}")
#
# # 可视化轮廓系数
# plt.plot(range(2, 11), silhouette_scores)
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette score')
# plt.show()

# 选择最优的K值
# optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 选择轮廓系数最大对应的K值
optimal_k = 2
print(f"The optimal k value is {optimal_k}")

# 进行聚类分析
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(data_scaled)
data['Cluster'] = kmeans.labels_

# 计算每个类别的平均值
cluster_means = data.groupby('Cluster').mean()

# 计算每个类别的相对重要性
cluster_importance = cluster_means.sum(axis=1) / cluster_means.sum().sum()

# 计算每个类别的相对价值，避免分母为0
cluster_value = cluster_means.divide(cluster_importance, axis=0)

# 绘制雷达图
categories = data.drop('Cluster', axis=1).columns
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

# 控制子图布局
fig, axs = plt.subplots(optimal_k, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'), gridspec_kw=dict(hspace=0.5))

for i, ax in enumerate(axs):
    values = cluster_value.iloc[i].values
    values = np.concatenate((values, [values[0]]))
    # values_std = (values - values.mean())/values.std()
    # # 根号
    # values_std = np.sqrt(values_std)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    args1 = angles * 180 / np.pi
    args2 = categories
    ax.set_thetagrids(args1, args2)
    ax.set_title(f"Cluster {i + 1}")
    ax.grid(True)

plt.show()