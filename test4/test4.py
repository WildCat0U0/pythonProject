import pandas as pd
import csv

data_start = pd.read_csv('air_data.csv')

# 构建LRFMC模型
data_start['L'] = data_start['LOAD_TIME'] - data_start['FFP_DATE']
data_start['R'] = data_start['LAST_TO_END']
data_start['F'] = data_start['FLIGHT_COUNT']
data_start['M'] = data_start['SEG_KM_SUM']
data_start['C'] = data_start['avg_discount']

# 构建其他属性
data_start['A'] = data_start['age'] # 年龄
data_start['B'] = data_start['BP_SUM'] # 总的基本积分
data_start["S1"] = data_start['SUM_YR_1'] # 第一年总票价
data_start["Lfd"] = data_start['LAST_FLIGHT_DATA'] # 最后一次飞行时间

# 删除原有属性
data_start.drop(['FFP_DATE', 'LOAD_TIME', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount', 'age', 'BP_SUM', 'SUM_YR_1', 'LAST_FLIGHT_DATA'], axis=1, inplace=True)


data = data_start["MEMBER_NO", "L", "R", "F", "M", "C", "A", "B", "S1", "Lfd"]
data.to_csv("data.csv", index=False, sep=',', encoding='utf-8')
