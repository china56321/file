# -*- coding:UTF-8 -*-
#!/usr/bin/python2.7
import cv2
import argparse
import os
import sys
import numpy as np
path='/home/ambavm/video_to_img/video/ts/ts/'
VideoPath = os.listdir(path)

j=1
def video_to_img():
	for video_name in VideoPath:
         
         image_file = path + video_name

         print(image_file)
         # 使用opencv按一定间隔截取视频帧，并保存为图片
         vc = cv2.VideoCapture(image_file) #读取视频文件
         if vc.isOpened():#判断是否正常打开
            flag,frame = vc.read()        
         else:        
             flag = False        
         c = 1
         d=0
         timeF = 30 #视频帧计数间隔频率
         while flag: #循环读取视频帧
              flag,frame = vc.read()
              if frame is not None :
                 if (c%timeF == 0):  #每隔timeF帧进行存储操作
                    cv2.imwrite(str(c)+'.jpg',frame) # 存储为图像
                    print("*************  Number ---> %s *************** " % (d))
                    d+=1
                    

              c = c + 1
              cv2.waitKey(1)
         vc.release()
         global j
         print("*************************")
         rename()
         j+=1


def rename():
    filelist = os.listdir(path)
    total_num = len(filelist)
    i = 0 #图片名称起始值

    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)
            str1=str(i)
            dst = os.path.join(os.path.abspath(path), str1.zfill(j) +'_tunnel'+'.jpg') #图片名称为5位数，即第一张图片为000000，第二张图片为000001，以此类推.......
            try:
                os.rename(src, dst)
                print ('converting %s to %s ...' % (src, dst))
                i+= 1
            except:
                  continue
   
       # print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
     video_to_img()
