import pandas as pd

data_start = pd.read_csv('air_data.csv')

# 构建LRFMC模型
data_start['LOAD_TIME'] = pd.to_datetime(data_start['LOAD_TIME'])
data_start['FFP_DATE'] = pd.to_datetime(data_start['FFP_DATE'])
data_start['L'] = (data_start['LOAD_TIME'] - data_start['FFP_DATE']).dt.days / 30 # 会员入会时间距观测窗口结束的月数
data_start['R'] = data_start['LAST_TO_END']
data_start['F'] = data_start['FLIGHT_COUNT']
data_start['M'] = data_start['SEG_KM_SUM']
data_start['C'] = data_start['avg_discount']

# 构建其他属性
data_start['A'] = data_start['AGE'] # 年龄
data_start['B'] = data_start['BP_SUM'] # 总的基本积分
data_start["S1"] = data_start['SUM_YR_1'] # 第一年总票价
data_start["Lfd"] = data_start['LAST_FLIGHT_DATE'] # 最后一次飞行时间

# 删除原有属性
#data_start.drop(['FFP_DATE', 'LOAD_TIME', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount', 'AGE', 'BP_SUM', 'SUM_YR_1', 'LAST_FLIGHT_DATE'], axis=1, inplace=True)

# 处理缺失值
data_start = data_start[data_start['L'].notnull() & data_start['R'].notnull() & data_start['F'].notnull() &data_start['M'].notnull() & data_start['C'].notnull() & data_start['A'].notnull() & data_start['B'].notnull() & data_start['S1'].notnull() & data_start['Lfd'].notnull()]

# 处理异常值
data_start.dropna(inplace=True)
data = data_start[["MEMBER_NO", "L", "R", "F", "M", "C", "A", "B", "S1", "Lfd"]]
data.to_csv("./data.csv", index=False, sep=',', encoding='utf-8')