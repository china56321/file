# -*- coding: utf-8 -*-
'''挑选文件夹下图片的对应xml文件'''

import os
#import os.path
import shutil  #Python文件复制相应模块
 
label_dir='/home/ambavm/BDD/train-xml-ok'  #所有xml文件所在文件夹
annotion_dir='/home/ambavm/BDD/choose_test_xml'  #粘贴对应图片名称的xml文件到指定文件夹
path = '/home/ambavm/BDD/choose_test_img'   #图片文件夹
path_list = os.listdir(path)# os.listdir(file)会历遍文件夹内的文件并返回一个列表
print(path_list)
path_name=[]  # 定义一个空列表,不需要path_list中的后缀名
# 利用循环历遍path_list列表并且利用split去掉后缀名
for i in path_list:
    path_name.append(i.split(".")[0])
#print(path_name)
# 排序一下
path_name.sort()
for file_name in path_name:
    # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    with open("save.txt","a") as f:
        f.write(file_name + "\n")
    f.close()
f = open("save.txt","r")   #设置文件对象
lines= f.readlines() 
s=[]
i=0
for line in lines:
    line = line.strip() 
    tempxmlname='%s.xml'%line
    xmlname=os.path.join(label_dir,tempxmlname)
    print (xmlname)
    os.listdir(label_dir) 
    shutil.move(xmlname,annotion_dir)
    i+=1
    print(i)
