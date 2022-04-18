# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import cv2
import numpy as np
import os
import json
from matplotlib import pyplot as plt
# 如何获取颜色特征
class ColorDescriptor:
    def __init__(self, bins):
        # 存储 3D 直方图的数量
        self.bins = bins

    def describe(self, image):
        # 将图像转换为 HSV 色彩空间并初始化
        # HSV 色彩空间：H 0-179，S 0-255，V 0-255
        # BGR: 0-255
        # 用于量化图像的特征
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)  # OpenCV读取颜色顺序：BRG
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # print("image", image)
        features = []
        # 获取尺寸并计算图像的中心
        (h, w) = image.shape[:2] # height, width
        (cX, cY) = (int(w * 0.5), int(h * 0.5)) # half of height, half of width
        # print("cx, cy", (cX, cY))
        # 将图像分成四份 rectangles/segments (top-left,top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
        # 构建代表图像中心的椭圆蒙版
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)  # 椭圆轴心
        ellipMask = np.zeros(image.shape[:2], dtype="uint8") # 先构造整张图片的向量
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1) # 中心椭圆蒙版
        # loop over the segments
        for (startX, endX, startY, endY) in segments:  # 图像的四份
            # 为图像的每个角构建一个掩码，从中减去椭圆中心
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1) # 四份各自的蒙版
            cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
            # 从图像中提取颜色直方图，然后更新特征向量
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        # 从椭圆区域提取颜色直方图并更新特征向量
        # 共有五份特征向量，四个角 + 椭圆
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        # 返回特征向量
        return features

    def histogram(self, image, mask):
        # 使用提供的每个通道的 bin 数量，从图像的遮罩区域中提取 3D 颜色直方图
        # bins = [8, 12, 3], 第二个参数是计算的通道数，mask掩码
        # bins 是对三个颜色的分割，实现降维；例如[16,16,16] 最终分割 256 / 16 = 16. 16 * 16 * 16 = 4096 种像素index
        # bins 实现降维 self.bins

        # hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.calcHist([image], [0, 1, 2], mask, (16, 16, 16), [0, 256, 0, 256, 0, 256])  # HSV
        hist = cv2.normalize(hist, hist).flatten() # 归一化
        return hist
        # 返回直方图



def distance(histA, histB, eps = 1e-10):
    # 计算卡方距离， 越小表示越相似
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
    # 返回卡方距离
    return d

def BhattachAraay(histA, histB):
    # 计算巴氏距离 ，范围在0 - 1 之间，越小表示越相似
    match = cv2.compareHist(histA, histB, cv2.HISTCMP_BHATTACHARYYA)
    return match


def get_feature(img_file, bins):
    cd = ColorDescriptor(bins)
    test_img_ = cv2.imread(img_file)
    test_img_features = np.array(cd.describe(test_img_))
    return test_img_features


def pic_library(piclib, pic_path):
    pic_list = os.listdir(pic_path)
    lib = open(piclib, "w")
    for pic in pic_list:
        path_p = pic_path + pic
        tmp_features = list(get_feature(path_p, bins))
        features = [str(f) for f in tmp_features]
        lib.write("%s,%s\n" % (path_p, ",".join(features)))
    lib.close()


def get_sift_features(pic_path):
    numWords = 64
    sift_detector = cv2.SIFT_create()
    des_list =[]
    pic_list = os.listdir(pic_path)
    for pic in pic_list:
        path = pic_path + pic
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift_detector.detectAndCompute(gray, None)  #
        des_list.append(des)


def query_pic(lib, query_image):
    query_img_features = get_feature(query_image, bins)
    f = open(lib, "r", encoding="utf-8")
    reader = csv.reader(f)
    dis_dic = {}
    for pic in reader:
        tmp_features = [float(x) for x in pic[1:]]  # pic[0] 是 图像文件名
        d = distance(tmp_features, query_img_features)
        dis_dic[pic[0]] = d
    result = sorted([(v, k) for (k, v) in dis_dic.items()])
    print(result)


def query_direct(allpic, image):
    query_img_features = get_feature(image, bins)
    pic_p = os.listdir(allpic)
    pic_dic = {}
    for item in pic_p:
        path_tmp = allpic + item
        tmp_features = get_feature(path_tmp, bins)
        d = BhattachAraay(tmp_features, query_img_features)
        # print(path_tmp, d)
        pic_dic[path_tmp] = d
    result = sorted([(v,k) for (k,v) in pic_dic.items()])
    print(result)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    test_img = "./snowmountain_test.png"

    features_lib = "all_features.csv"
    all_pictures_dir = "./pic_library/"
    # bins = (8, 12, 3)  # HSV bins
    bins = (16,16,16)  # BGR bins

    pic_library(features_lib, all_pictures_dir)
    query_pic(features_lib, test_img)
    query_direct(all_pictures_dir, test_img)

