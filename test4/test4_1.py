import pandas as pd
import time

data = pd.read_csv('data.csv')
# 处理缺失值

# 将最后一次飞行时间转换为时间戳
data['Lfd'] = data['Lfd'].replace("2014/2/29  0:00:00", "2014/2/28")
data['Lfd'] = data['Lfd'].apply(lambda x: time.mktime(time.strptime(x, '%Y/%m/%d')))

# 将时间戳转换为以天为单位的整数
data['Lfd'] = data['Lfd'].apply(lambda x: int(x // 86400 - 10950)) #从1970年算起，到现在的天数
# 1970 - 2000 年是 30 * 365 = 10950 天

data.to_csv("./data.csv", index=False, sep=',', encoding='utf-8')


