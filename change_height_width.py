# -*- coding:UTF-8 -*-
#获取文件夹中的文件路径
import os
def getFilePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list

#修改文件夹中的单个xml文件
import xml.etree.ElementTree as ET
def single_xmlCompress(old_xmlFilePath, new_xmlFilePath):
    with open(old_xmlFilePath) as file:
         fileContent = file.read()
    root = ET.XML(fileContent)
    #获得图片宽度变化倍数，并改变xml文件中width节点的值
    width = root.find('size').find('width')
    a=960
    width.text=a
    old_width = int(width.text)
    width.text = str(old_width)
    #获得图片高度变化倍数，并改变xml文件中height节点的值
    height = root.find('size').find('height')
    b=540
    height.text=b
    old_height = int(height.text)
    height.text = str(old_height)
 
    tree = ET.ElementTree(root)
    tree.write(new_xmlFilePath)
    
#修改文件夹中的若干xml文件
def batch_xmlCompress(old_dirPath, new_dirPath):
    xmlFilePath_list = getFilePathList(old_dirPath, '.xml')
    for xmlFilePath in xmlFilePath_list:
        old_xmlFilePath = xmlFilePath
        xmlFileName = os.path.split(old_xmlFilePath)[1]
        new_xmlFilePath = os.path.join(new_dirPath, xmlFileName)
        single_xmlCompress(xmlFilePath, new_xmlFilePath)

# 主函数    
if __name__ == '__main__':
    batch_xmlCompress('/home/share/make/darknet_self_mark/darknet/VOCdevkit/VOC2007/Annotations/', '/home/share/make/darknet_self_mark/darknet/VOCdevkit/VOC2007/b/')
    print('all xml files have been compressed')
