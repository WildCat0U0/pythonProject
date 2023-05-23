import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#  读取数据集
train = pd.read_csv("train.csv")
train.set_index('Loan_ID', inplace=True)
test = pd.read_csv("test.csv")
test1 = test
# test.set_index('Loan_ID', inplace=True)



train["Gender"] = train["Gender"].map({'Female': 0, 'Male': 1})
test["Gender"] = test["Gender"].map({'Female': 0, 'Male': 1})
train["Married"] = train["Married"].map({'No': 0, 'Yes': 1})
train["Education"] = train["Education"].map({'Not Graduate': 0, 'Graduate': 1})
train["Self_Employed"] = train["Self_Employed"].map({'No': 0, 'Yes': 1})
train["Property_Area"] = train["Property_Area"].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
train["Loan_Status"] = train["Loan_Status"].map({'N': 0, 'Y': 1})
test["Married"] = test["Married"].map({'No': 0, 'Yes': 1})
test["Education"] = test["Education"].map({'Not Graduate': 0, 'Graduate': 1})
test["Self_Employed"] = test["Self_Employed"].map({'No': 0, 'Yes': 1})
test["Property_Area"] = test["Property_Area"].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})

# denpendet 3+
train["Dependents"] = train["Dependents"].replace('3+', 3)
test["Dependents"] = test["Dependents"].replace('3+', 3)

# 查看数据集的缺失值情况并处理
imputer = SimpleImputer(strategy="mean")
train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)

#  查看数据集的缺失值情况并处理
test.dropna(subset=["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"],inplace=True)
new_df = pd.DataFrame(test['Loan_ID'])
test1 = test.copy()
test.set_index('Loan_ID', inplace=True)


#  对类别变量进行编码
train = pd.get_dummies(train, drop_first=True)
# test = pd.get_dummies(test, drop_first=True)

#  定义特征变量和目标变量
X_train = train.drop("Loan_Status", axis=1)
y_train = train["Loan_Status"]
X_test = test

#  划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

#  建立logistic回归模型，并使用交叉验证评估准确率，使用网格搜索来寻找最佳参数组合
logreg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
grid_search_logreg = GridSearchCV(logreg, param_grid=param_grid, cv=5)
grid_search_logreg.fit(X_train, y_train)
logreg_best = grid_search_logreg.best_estimator_
scores = cross_val_score(logreg_best, X_train, y_train, cv=5)
print("Logistic  Regression  Accuracy:  %0.2f  (+/-  %0.2f)" % (scores.mean(), scores.std() * 2))

#  在验证集上预测并打印混淆矩阵和分类报告
y_pred_logreg = logreg_best.predict(X_val)
print(confusion_matrix(y_val, y_pred_logreg))
print(classification_report(y_val, y_pred_logreg))

#  建立决策树模型，并使用交叉验证评估准确率，使用网格搜索来寻找最佳参数组合
dtree = DecisionTreeClassifier()
param_grid = {'max_depth': [2, 4, 6, 8]}
grid_search_dtree = GridSearchCV(dtree, param_grid=param_grid, cv=5)
grid_search_dtree.fit(X_train, y_train)
dtree_best = grid_search_dtree.best_estimator_
scores = cross_val_score(dtree_best, X_train, y_train, cv=5)
print("Decision  Tree  Accuracy:  %0.2f  (+/-  %0.2f)" % (scores.mean(), scores.std() * 2))

#  在验证集上预测并打印混淆矩阵和分类报告
y_pred_dtree = dtree_best.predict(X_val)
print(confusion_matrix(y_val, y_pred_dtree))
print(classification_report(y_val, y_pred_dtree))

#  建立随机森林模型，并使用交叉验证评估准确率，使用网格搜索来寻找最佳参数组合
rf = RandomForestClassifier(n_estimators=100, random_state=0)
param_grid = {'max_depth': [2, 4, 6, 8], 'min_samples_split': [2, 4, 6], 'min_samples_leaf': [1, 2, 3]}
grid_search_rf = GridSearchCV(rf, param_grid=param_grid, cv=5)
grid_search_rf.fit(X_train, y_train)
rf_best = grid_search_rf.best_estimator_
scores = cross_val_score(rf_best, X_train, y_train, cv=5)
print("Random  Forest  Accuracy:  %0.2f  (+/-  %0.2f)" % (scores.mean(), scores.std() * 2))

#  在验证集上预测并打印混淆矩阵和分类报告
y_pred_rf = rf_best.predict(X_val)
print(confusion_matrix(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))

# # 给test数据集给予预测
# y_pred_test = rf_best.predict(X_test)
# test["Loan_Status"] = y_pred_test
# test["Loan_Status"] = test["Loan_Status"].map({0: 'N', 1: 'Y'})
# test[["Loan_ID", "Loan_Status"]].to_csv("submission.csv", index=False)


#    给测试集进行预测并将结果保存
# print(test.head(5))
# a = []
# a = rf_best.predict(X_test)
# print(a)
print(test1.head(5))

test1["Loan_Status"] = rf_best.predict(X_test)  # 用模型预测结果
test1["Loan_Status"] = test1["Loan_Status"].apply(lambda x: 'Y' if x == 1 else 'N')  # 将结果转换为Y和N
pd.DataFrame(test1, columns=["Loan_ID", "Loan_Status"]).to_csv('submission.csv', index=False)  # 保存结果
# test.set_index("Loan_ID")[["Loan_Status"]].to_csv("submission.csv")
# test[["Loan_ID", "Loan_Status"]].to_csv("submission.csv", index=False)
# pd.DataFrame(test["Loan_Status"]).to_csv('submission.csv', index=False)

# #    将结果保存为csv文件
# test.set_index("Loan_ID")[["Loan_Status"]].to_csv("submission.csv")
# test[["Loan_ID", "Loan_Status"]].to_csv("submission.csv", index=False)  # 设置index=False，避免输出文件中多余的行号信息
