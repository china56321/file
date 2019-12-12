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

    #获取标记物体的列表，修改其中xmin,ymin,xmax,ymax这4个节点的值
    object_list = root.findall('object')
    for object_item in object_list:
        object_item=object_item.find('name')
        if object_item.text =="motorcycle":
           object_item.text = "motorbike"
        # elif object_item.text =="bicycle":
        #     object_item.text = "bike"

    tree = ET.ElementTree(root)
    tree.write(new_xmlFilePath)
    
#修改文件夹中的若干xml文件
def batch_xmlCompress(old_dirPath, new_dirPath):
    i=0
    xmlFilePath_list = getFilePathList(old_dirPath, '.xml')
    for xmlFilePath in xmlFilePath_list:
        old_xmlFilePath = xmlFilePath
        xmlFileName = os.path.split(old_xmlFilePath)[1]
        new_xmlFilePath = os.path.join(new_dirPath, xmlFileName)
        single_xmlCompress(xmlFilePath, new_xmlFilePath)
        i+=1
        print(i)

# 主函数    
if __name__ == '__main__':
    batch_xmlCompress('/home/lifan/share/make/RefineDet/data/coco/VOC2007/Annotations/', '/home/lifan/share/make/RefineDet/data/coco/VOC2007/Annotations1/')
    print('all xml files have been compressed')
