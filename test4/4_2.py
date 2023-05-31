import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from datetime import datetime
import time

# 读取数据
data = pd.read_csv('data.csv')

# 选择需要的属性
# selected_features = ['L', 'R', 'F', 'M', 'C', 'A', 'B', 'S1', 'Lfd']
# data = data[selected_features]
data.set_index('MEMBER_NO', inplace=True)
data.dropna(inplace=True)
data["Lfd"] = data["Lfd"].replace("2014/2/29  0:00:00", "2014/2/28")
data['Lfd'] = data['Lfd'].apply(lambda x: time.mktime(time.strptime(x, '%Y/%m/%d')))
# 处理最后一次飞行日期
now = datetime.now()
data['Lfd'] = pd.to_datetime(data['Lfd'])
data['Lfd'] = (now - data['Lfd']) / np.timedelta64(1, 'M')
data['Lfd'] = data['Lfd'].astype(int)
# 数据预处理
# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)




# 寻找最优的K值
print("starting searching for the best K...")
# silhouette_scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
#     kmeans.fit(data_scaled)
#     score = silhouette_score(data_scaled, kmeans.labels_)
#     silhouette_scores.append(score)
#     print("success for k = ", k, "score = ", score)

# optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
optimal_k = 2
print(f"The optimal k value is {optimal_k}")

# 进行聚类分析
kmeans = KMeans(n_clusters=optimal_k, random_state=42,n_init=10)
kmeans.fit(data_scaled)
data['Cluster'] = kmeans.labels_

# 计算每个类别的平均值
cluster_means = data.groupby('Cluster').mean()  # 按照类别分组，计算每个类别的平均值

# 计算每个类别的相对重要性
cluster_importance = cluster_means.sum(axis=1) / cluster_means.sum().sum()

# 计算每个类别的相对价值
cluster_value = cluster_means.apply(lambda x: x / cluster_importance) # 每个类别的平均值除以每个类别的相对重要性

# 绘制雷达图
categories = data.drop('Cluster', axis=1).columns
print(categories)
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
print(angles)

i = 0

fig = plt.figure(figsize=(10, 10))  # 创建画布
for i in range(optimal_k):  # 画出每个类别的雷达图
    ax = fig.add_subplot(optimal_k, 1, i + 1, polar=True)  # 创建子图
    values = cluster_value.iloc[i].values  # 获取每个类别的相对价值
    values = np.concatenate((values, [values[0]]))  # 将第一个值复制到最后一个值，以便于画出闭合的雷达图
    print(values)
    # angles = np.concatenate((angles, angles[0]))
    ax.plot(angles, values, 'o-', linewidth=2)  # 画出雷达图
    ax.fill(angles, values, alpha=0.25)  # 填充颜色
    args1 = angles * 180 / np.pi
    args2 = values
    ax.set_thetagrids(args1, args2)  # 设置角度刻度
    ax.set_title(f"Cluster {i + 1}")  # 设置标题
    ax.grid(True)  # 设置网格线

plt.show()
