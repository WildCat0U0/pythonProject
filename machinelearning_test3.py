import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


df = pd.read_csv('shuizhi.csv')

# 分离测试集和训练集
X = df.iloc[:, 0:9]
y = df.iloc[:, 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#  放大系数k的选取：每个维度乘上一个大于1的系数k，提高数据区分度
k = 2
X_train = X_train * k
X_test = X_test * k

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

#  使用K折交叉验证
scores = cross_val_score(svc, X_train, y_train, cv=5)
print('SVC模型的准确率为：{:.2f}'.format(scores.mean()))
