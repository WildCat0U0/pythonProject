import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('./房屋数据.csv')

# 处理缺失值
data = data.dropna()

# 处理异常值（可根据具体情况进行异常值处理）
# 数据预处理
data.dropna(inplace=True)  # 处理缺失值
data = data[data['median_house_value'] < 500000]  # 处理异常值

# 将枚举类型转换成数值型
label_encoder = LabelEncoder()
data['ocean_proximity'] = label_encoder.fit_transform(data['ocean_proximity'])

# 属性规约（可根据具体情况进行属性规约）

# 数据归一化
scaler = StandardScaler()
data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
          'median_income']] = scaler.fit_transform(data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population','households',
          'median_income']])

# 分离特征和目标变量
X = data[['longitude', 'latitude', 'housing_median_age','total_rooms','total_bedrooms','population','households',
          'median_income', 'ocean_proximity']]
y = data['median_house_value']

features = ['longitude', 'latitude', 'housing_median_age','total_rooms','total_bedrooms','population','households',
            'median_income', 'ocean_proximity']
target = 'median_house_value'
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 定义回归模型
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor())
]

# 使用交叉验证评估模型
results = []
for name, model in models:
    pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    results.append((name, rmse_scores))

# 打印结果
for name, scores in results:
    print(f'{name}: Mean RMSE = {scores.mean()}, Standard Deviation = {scores.std()}')

# 选择最佳模型
best_model = models[np.argmin([scores.mean() for _, scores in results])][1]

# 在测试集上进行预测
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
#
# 计算预测结果的准确度（均方根误差 RMSE）
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE) on test set: {rmse}')

# 可选：可视化预测结果与真实值的比较
# plt.scatter(y_test, y_pred)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True Values vs Predictions')
# plt.show()


# cv_scores = []
# for name,model in models:
#     pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('model', model)])
#     scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
#     mse_scores = -scores  # 转换为正数
#     cv_scores.append(mse_scores)

# plt.boxplot(cv_scores, labels=[model.__class__.__name__ for model in models])
# # plt.title('Model Comparison')
# plt.title('模型比较图')
# plt.xlabel('模型')
# plt.ylabel('损失')
# plt.show()