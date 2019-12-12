# -*- coding:UTF-8 -*-
import cv2
# 使用opencv按一定间隔截取视频帧，并保存为图片
 
vc = cv2.VideoCapture('1.MP4') #读取视频文件

if vc.isOpened():#判断是否正常打开
 
    flag,frame = vc.read()
else:
    flag = False
 
c = 1
timeF = 3 #视频帧计数间隔频率
 
while  flag: #循环读取视频帧
     flag,frame = vc.read()
     if (c%timeF == 0):  #每隔timeF帧进行存储操作
        cv2.imwrite(str(c)+'.jpg',frame) # 存储为图像
 
     c = c + 1
     cv2.waitKey(1)
vc.release()
