# ml test3 图像处理部分
import cv2
n = 1
for i in range(51):

    path = "images/1_"+str(n)+".jpg"
    img = cv2.imread(path)
    height, width = img.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)
    crop_size = 100
    crop_x1 = center_x - int(crop_size / 2)
    crop_y1 = center_y - int(crop_size / 2)
    crop_x2 = center_x + int(crop_size / 2)
    crop_y2 = center_y + int(crop_size / 2)
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    path = "img/1_" + str(n) + ".jpg"
    cv2.imwrite(path,cropped_img)
    n = n + 1
    print("success" + path)
n = 1
for i in range(44):
    path = "images/2_" + str(n) + ".jpg"
    img = cv2.imread(path)
    height, width = img.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)
    crop_size = 100
    crop_x1 = center_x - int(crop_size / 2)
    crop_y1 = center_y - int(crop_size / 2)
    crop_x2 = center_x + int(crop_size / 2)
    crop_y2 = center_y + int(crop_size / 2)
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    path = "img/2_" + str(n) + ".jpg"
    cv2.imwrite(path, cropped_img)
    print("success" + path)
    n = n + 1

n = 1
for i in range(78):

    path = "images/3_" + str(n) + ".jpg"
    img = cv2.imread(path)
    height, width = img.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)
    crop_size = 100
    crop_x1 = center_x - int(crop_size / 2)
    crop_y1 = center_y - int(crop_size / 2)
    crop_x2 = center_x + int(crop_size / 2)
    crop_y2 = center_y + int(crop_size / 2)
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    path = "img/3_" + str(n) + ".jpg"
    cv2.imwrite(path, cropped_img)
    print("success" + path)
    n = n + 1
n=1
for i in range(24):
    path = "images/4_" + str(n) + ".jpg"
    img = cv2.imread(path)
    height, width = img.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)
    crop_size = 100
    crop_x1 = center_x - int(crop_size / 2)
    crop_y1 = center_y - int(crop_size / 2)
    crop_x2 = center_x + int(crop_size / 2)
    crop_y2 = center_y + int(crop_size / 2)
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    path = "img/4_" + str(n) + ".jpg"
    cv2.imwrite(path, cropped_img)
    print("success" + path)
    n = n + 1
n = 1
for i in range(6):
    path = "images/5_" + str(n) + ".jpg"
    img = cv2.imread(path)
    height, width = img.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)
    crop_size = 100
    crop_x1 = center_x - int(crop_size / 2)
    crop_y1 = center_y - int(crop_size / 2)
    crop_x2 = center_x + int(crop_size / 2)
    crop_y2 = center_y + int(crop_size / 2)
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    path = "img/5_" + str(n) + ".jpg"
    cv2.imwrite(path, cropped_img)
    print("success" + path)
    n = n + 1