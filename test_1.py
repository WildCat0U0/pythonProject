# 计算一阶矩，二阶矩，三阶矩
import cv2
import numpy as np
import csv

# heade# r = [['imgname',"b_mean","g_mean","r_mean", "b_2nd_moment", "g_2nd_moment", "r_2nd_moment", "b_3rd_moment", "g_3rd_moment", "r_3rd_moment"]]
# with open('shuizhi.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(header)
n=1
for i in range(6):
    path = "img/5_" + str(n) + ".jpg"
    img = cv2.imread(path)
    # 分离通道
    b, g, r = cv2.split(img)
    # 计算矩
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_2nd_moment = np.mean(np.square(b - b_mean))
    g_2nd_moment = np.mean(np.square(g - g_mean))
    r_2nd_moment = np.mean(np.square(r - r_mean))
    b_3rd_moment = np.mean(np.power(b - b_mean, 3))
    g_3rd_moment = np.mean(np.power(g - g_mean, 3))
    r_3rd_moment = np.mean(np.power(r - r_mean, 3))
    data = [[path,b_mean, g_mean, r_mean, b_2nd_moment, g_2nd_moment, r_2nd_moment, b_3rd_moment, g_3rd_moment, r_3rd_moment,5]]
    with open("shuizhi.csv",mode="a", newline="")as file:
        writer = csv.writer(file)
        writer.writerows(data)
    n = n+1

