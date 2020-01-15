# -*- coding:utf8 -*-

import cv2
import os

fullfilename=[]
filepath = "/home/delete/person/" #调整前图片的路径
filepath1 = "/home/delete/image_resize"  #调整后图片的路径

for filename in os.listdir(filepath):
    print(filename)
    print(os.path.join(filepath, filename))
    filelist = os.path.join(filepath, filename)
    fullfilename.append(filelist)


for imagename in fullfilename:
    img = cv2.imread(imagename)
    img = cv2.resize(img, (960, 540)) # 将图片尺寸调整为608*608
    a=imagename.split('.')[0][-6:]
    resizename = a+'.jpg'      
    isExists = os.path.exists(filepath1)
    if not isExists:
        os.makedirs(filepath1)
        print('mkdir resizename accomplished')
    savename = filepath1+'/'+resizename
    cv2.imwrite(savename, img)
    print('{} is resized'.format(savename))
   
