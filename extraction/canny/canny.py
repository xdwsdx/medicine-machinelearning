# coding: utf-8
import cv2
import os
import matplotlib.pyplot as plt
#********************Canny边缘检测*****************************
def edge_canny(src, threshold1, threshold2 ):
    grayImg = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)                 # 将图像转化为灰度图像
    kernelSize = (3, 3)                                             # 设置卷积核大小
    gausBlurImg = cv2.GaussianBlur(grayImg, kernelSize, 0 )         # 高斯滤波
    cannyImg = cv2.Canny( gausBlurImg, threshold1, threshold2 )
    return cannyImg

#********************主函数*****************************
if __name__ == '__main__':
    images = os.listdir('./images/')
    for image in images:
        imgSrc = cv2.imread( "./images/"+image )
        cannyImg = edge_canny( imgSrc, 20, 30 )
        cv2.imwrite('./features/'+image, cannyImg)