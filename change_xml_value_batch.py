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
    old_width = int(width.text)/2
    width.text = str(old_width)
    #获得图片高度变化倍数，并改变xml文件中height节点的值
    height = root.find('size').find('height')
    old_height = int(height.text)/2
    height.text = str(old_height)
    #获取标记物体的列表，修改其中xmin,ymin,xmax,ymax这4个节点的值
    object_list = root.findall('object')
    for object_item in object_list:
        bndbox = object_item.find('bndbox')
        xmin = bndbox.find('xmin')
        xminValue = int(xmin.text)/2
        xmin.text = str(int(xminValue))
        ymin = bndbox.find('ymin')
        yminValue = int(ymin.text)/2
        ymin.text = str(int(yminValue))
        xmax = bndbox.find('xmax')
        xmaxValue = int(xmax.text)/2
        xmax.text = str(int(xmaxValue))
        ymax = bndbox.find('ymax')
        ymaxValue = int(ymax.text)/2
        ymax.text = str(int(ymaxValue))
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
    batch_xmlCompress('/home/lifan/test/pxml/', '/home/lifan/test/tested')
    print('all xml files have been compressed')
