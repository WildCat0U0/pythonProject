import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns

# 机器学习 第三周实验代码
# submission.csv # 用于存放预测结果
# shuizhi.csv # 用于存放训练数据
# test_1.py 用于处理数据
# test3.py  用于处理数据
#  读入数据
df = pd.read_csv('shuizhi.csv')

#  分离测试集和训练集
X = df.iloc[:, 0:9]
y = df.iloc[:, 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#  数据归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  进行参数优化
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('最佳参数为：', grid_search.best_params_)

#  创建SVC模型
svc = SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'])

#  用训练集训练模型
svc.fit(X_train, y_train)

#  用测试集预测
y_pred = svc.predict(X_test)

#  使用交叉验证评估模型
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(svc, X_train, y_train, cv=5, scoring=scoring)
print('SVC模型的准确率为：{:.2f}'.format(scores['test_accuracy'].mean()))
print('SVC模型的精度为：{:.2f}'.format(scores['test_precision_macro'].mean()))
print('SVC模型的召回率为：{:.2f}'.format(scores['test_recall_macro'].mean()))
print('SVC模型的F1值为：{:.2f}'.format(scores['test_f1_macro'].mean()))

#  混淆矩阵可视化
sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, cmap='YlGnBu') # 画混淆矩阵图
plt.title('Confusion Matrix')  # 图标题
plt.xlabel('Predicted Labels')  # x、y轴标题
plt.ylabel('True Labels')   # x、y轴标题
plt.show()

#  分类报告
print(classification_report(y_test, y_pred))

#  使用K折交叉验证
scores = cross_val_score(svc, X_train, y_train, cv=5)
print('SVC模型的准确率为：{:.2f}'.format(scores.mean()))
