# -*- coding : UTF-8 -*-
import os
import pywt
import cv2
import matplotlib.pyplot as plt

images_list = os.listdir('./images/')
for image in images_list:
    img = os.path.join('./images', image)
    img = cv2.imread(img)
    # 将多通道图像变为单通道图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    plt.subplot(221), plt.imshow(cA, 'gray'), plt.title("A")
    plt.subplot(222), plt.imshow(cH, 'gray'), plt.title("H")
    plt.subplot(223), plt.imshow(cV, 'gray'), plt.title("V")
    plt.subplot(224), plt.imshow(cD, 'gray'), plt.title("D")
    plt.show()

