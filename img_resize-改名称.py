# -*- coding:utf8 -*-

import cv2
import os

fullfilename=[]
filepath = "/home/lifan/share/ML/image_rename/" #调整前图片的路径
filepath1 = "/home/lifan/share/ML/image_resize"  #调整后图片的路径

for filename in os.listdir(filepath):
    print(filename)
    print(os.path.join(filepath, filename))
    filelist = os.path.join(filepath, filename)
    fullfilename.append(filelist)


i = 0
for imagename in fullfilename:
    img = cv2.imread(imagename)
    img = cv2.resize(img, (960, 540)) 
    resizename = str(i)+'.jpg'      
    isExists = os.path.exists(filepath1)
    if not isExists:
        os.makedirs(filepath1)
        print('mkdir resizename accomplished')
    savename = filepath1+'/'+resizename
    cv2.imwrite(savename, img)
    print('{} is resized'.format(savename))
    i = i+1
