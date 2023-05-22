# 导入必要的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 读取数据集
train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")

# 查看数据集的基本信息
# train.info()
# # test.info()
#
# # 查看数据集的统计描述
# train.describe()
# test.describe()

# 查看数据集的缺失值情况
train.isnull().sum()
# test.isnull().sum()

# 对缺失值进行处理，可以使用均值、中位数、众数或其他方法填充，也可以删除含有缺失值的行或列，这里以均值填充为例
train.dropna(axis=0, inplace=True)
# test.fillna(test.mean(), inplace=True)

# # 对类别变量进行编码，可以使用LabelEncoder或OneHotEncoder等方法，这里以OneHotEncoder为例
# train = pd.get_dummies(train, drop_first=True)
# # test = pd.get_dummies(test, drop_first=True)

# 查看编码后的数据集
train.head()
# test.head()

# 定义特征变量和目标变量，这里以Loan_ID为索引，Loan_Status为目标变量，其他列为特征变量
X_train = train.drop(["Loan_ID", "Loan_Status"], axis=1)
y_train = train["Loan_Status"]
# X_test = test.drop(["Loan_ID"], axis=1)

# 划分训练集和验证集，这里以8/2分，并设置随机种子为0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# 建立logistic回归模型，并使用交叉验证评估准确率，这里以5折交叉验证为例
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
scores = cross_val_score(logreg, X_train, y_train, cv=5)
print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 在验证集上预测并打印混淆矩阵和分类报告
# y_pred_logreg = logreg.predict(X_val)
# print(confusion_matrix(y_val, y_pred_logreg))
# print(classification_report(y_val, y_pred_logreg))

# 建立决策树模型，并使用交叉验证评估准确率，这里以5折交叉验证为例
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
scores = cross_val_score(dtree, X_train, y_train, cv=5)
print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 在验证集上预测并打印混淆矩阵和分类报告
# y_pred_dtree = dtree.predict(X_val)
# print(confusion_matrix(y_val, y_pred_dtree))
# print(classification_report(y_val, y_pred_dtree))

# 建立随机森林模型，并使用交叉验证评估准确率，这里以5折交叉验证为例
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X_train, y_train, cv=5)
print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 在验证集上预测并打印混