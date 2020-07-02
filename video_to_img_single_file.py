 # -*- coding: utf-8 -*-
 """ 
 该脚本用于将视频转换为图片。对每个视频文件新建一个文件夹，用于存放转换后的图片。
 """
import cv2
from skimage import io
import os
 # 视频所在文件夹的路径位置
videos_path = '/home/ambavm/video_to_img/video/ts/video/'
 # 从视频文件夹中获取各个视频的名字
videos_name = os.listdir(videos_path)
 # 视频帧图片的起始编号
 # 遍历所有的视频
for j, i in enumerate(videos_name):
    # 以下两行代码的作用是为每段视频帧创建一个文件夹
    # path = '/home/ambavm/video_to_img/video/ts/video/videos_name_%s'%(j+1)
    video_name=i.split(".")[0]
    path = '/home/ambavm/video_to_img/video/ts/video/%s'%(video_name)
    print(path)
    os.makedirs(path)
    # 获取每个视频的路径
    video_path = os.path.join(videos_path, i)
    # 如果遇见文件夹就跳过
    if os.path.isdir(video_path):
        continue
    vc = cv2.VideoCapture(video_path) # 参数0表示第一个摄像头

    #判断视频是否打开
    if vc.isOpened():
        rval,frame=vc.read()
        print('Open')
    else:
        rval=False
        print('UnOpen')
    
    # 测试用,查看视频size
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('size:'+repr(size))
    a=1
    c = 1
    timeF=30
    
    while rval:#循环读取视频帧
        rval,frame = vc.read( )
        if frame is not None:
            if(c%timeF == 0): #每隔timeF帧进行存储操作
                cv2.imwrite("/home/ambavm/video_to_img/video/ts/video/%s/%s.jpg"%(video_name,str(a)+"_zz"),frame) #存储为图像
                print("*************  Number ---> %s *************** " % (a))
                a=a+1
            c=c+1
            cv2.waitKey(1)
    vc.release()
    
cv2.destroyAllWindows()
