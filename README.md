# **Content_Image**

## **0 Affine Transformation**
**基于OpenCV实现仿射变换**  
* Affine.ipynb   
  仿射变换，包含平移、旋转、缩放、翻转、错切的实现，基于getAffineTransform和warpAffine函数；含有可视化对比。

---

## **1 Color_Hist_Features**
**基于颜色直方图特征的图像匹配**  
* /pic_library/，是图像库，可自行构建图像素材库 
* main.py， 提取颜色直方图特征，使用calHist函数
* **hist_features.ipynb**，提取颜色直方图特征，查询图片，构造图像特征库，分步调试
* Color_Hist.ipynb，RGB/HSV颜色直方图调试理解
* all_features.csv，根据图像库和特征提取算法，获得的图像特征总文件，多次查询时，可避免重复计算特征  

---

## **2 HOG+SVM_INRIAData**
* **HOG_SVM_INRIAData.ipynb**，使用HOG Descriptor 提取数据集特征，训练SVM分类模型，测试。  

**INRIA 行人数据集**，已进行修复和尺寸规范化，训练集（正例2416条，负例1218条），测试集（正例1126条），[Google Drive下载](https://drive.google.com/file/d/1peZ-uRV9JDMDOXTseQMi2izBWDvMNg0x/view?usp=sharing)  
**SVM训练模型**，测试集准确率为100%，[Google Drive下载](https://drive.google.com/file/d/1MD7lGp-VVzVwPEZsSmGbA-wjcdyWmW-A/view?usp=sharing ) 

---

## **3 SIFT**
重在理解SIFT算法，如何实现尺度空间不变性


---


## **4 YOLOv3**
* **yolov3-test.ipynb**，YOLOv3目标检测，输出三个尺度的检测结果。

需下载YOLOv3权值、配置文件、标签名，yolov3.weights、yolov3.cfg、coco.names 三个文件，放在该目录下。  

**YOLOv3配置文件** [Google Drive下载](https://drive.google.com/drive/folders/1jUE0ajKH79eooPZVstuipgSmiczTqSEV?usp=sharing)

