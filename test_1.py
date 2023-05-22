# import the necessary packages
import numpy as np
import cv2
import os
import csv
n = 1
for i in range(6):
    path = "img/5_"+str(n)+".jpg"
    img = cv2.imread(path)
    # Convert BGR to HSV colorspace
    # Split the channels - h,s,v
    b,g,r = cv2.split(img)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    b_mean = np.mean(b)/255 # np.sum(h)/float(N)
    g_mean = np.mean(g)/255  # np.sum(s)/float(N)
    r_mean = np.mean(r)/255  # np.sum(v)/float(N)
    color_feature.extend([b_mean, g_mean, r_mean])
    # The second central moment - standard deviation
    b_std = np.std(b)/255  # np.sqrt(np.mean(abs(h - h.mean())**2))
    g_std = np.std(g)/255  # np.sqrt(np.mean(abs(s - s.mean())**2))
    r_std = np.std(r)/255  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([b_std, g_std, r_std])
    # The third central moment - the third root of the skewness
    b_skewness = np.mean(abs(b - b.mean())**3)
    g_skewness = np.mean(abs(g - g.mean())**3)
    r_skewness = np.mean(abs(r - r.mean())**3)
    b_thirdMoment = b_skewness**(1./3)
    g_thirdMoment = g_skewness**(1./3)
    r_thirdMoment = r_skewness**(1./3)
    color_feature.extend([b_thirdMoment/255, g_thirdMoment/255, r_thirdMoment/255])
    color_feature.extend([5])
    with open("shuizhi.csv",mode="a",newline="")as file:
        writer = csv.writer(file)
        writer.writerows([color_feature])
    print("success" + str(n) + "file")
    n = n + 1
