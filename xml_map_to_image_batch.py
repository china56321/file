# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
  
def read_xml(ImgPath, AnnoPath, Savepath):  
    i = 1  
    imagelist = os.listdir(AnnoPath)  
    for image in imagelist:  
        image_pre, ext = os.path.splitext(image)  
        imgfile = ImgPath + '/' + image_pre + '.jpg'  
        xmlfile = AnnoPath + '/'  + image_pre + '.xml'  
        im = cv2.imread(imgfile)  
        DomTree = xml.dom.minidom.parse(xmlfile)  
        annotation = DomTree.documentElement  
        filenamelist = annotation.getElementsByTagName('filename')  
        filename = filenamelist[0].childNodes[0].data  
        objectlist = annotation.getElementsByTagName('object')  
        
        for objects in objectlist:  
            namelist = objects.getElementsByTagName('name')  
            objectname = namelist[0].childNodes[0].data  
            bndbox = objects.getElementsByTagName('bndbox')  
  
            for box in bndbox:  
                try:  
                    x1_list = box.getElementsByTagName('xmin')  
                    x1 = int(x1_list[0].childNodes[0].data)  
                    y1_list = box.getElementsByTagName('ymin')  
                    y1 = int(y1_list[0].childNodes[0].data)  
                    x2_list = box.getElementsByTagName('xmax')  
                    x2 = int(x2_list[0].childNodes[0].data)  
                    y2_list = box.getElementsByTagName('ymax')  
                    y2 = int(y2_list[0].childNodes[0].data)  
  
                    minX = x1  
                    minY = y1  
                    maxX = x2  
                    maxY = y2  
  
                    if(i % 9 == 0):  
                        color = (255,97,0)  
                    elif (i % 9 == 1):  
                        color = (255, 255, 0)  
                    elif (i % 9 == 2):  
                        color = (65, 105, 255)  
                    elif (i % 9 == 3):  
                        color = (0, 255, 255)  
                    elif (i % 9 == 4):  
                        color = (8, 46, 84)  
                    elif (i % 9 == 5):  
                        color = (0, 255, 0)  
                    elif (i % 9 == 6):  
                        color = (0, 255, 255)  
                    elif (i % 9 == 7):  
                        color = (255, 0, 0)  
                    elif (i % 9 == 8):  
                        color = (255, 0, 255)  
                    elif (i % 9 == 9):  
                        color = (0, 199, 140)  
                    cv2.rectangle(im,(minX,minY),(maxX,maxY),color,2)  
  
#                    path = Savepath + '/' + image_pre + '_New' + '.jpg'  
                    path = Savepath + '/' + image_pre + '.jpg'  
                    font = cv2.FONT_HERSHEY_SIMPLEX  
                    cv2.putText(im, objectname, (minX,minY - 7), font, 0.5, (0, 0, 255), 1)  
                    cv2.imwrite(path, im)  
  
  
                    i += 1  
                    print(i)
                except Exception as e:  
                    print (e)
 
read_xml('JPEGImages','Annotations','save')

























