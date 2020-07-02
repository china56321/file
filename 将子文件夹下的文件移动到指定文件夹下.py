# -*- coding:UTF-8 -*-
import shutil,os
new_path='/home/lifan/share/ML/img1' #移动后的文件夹路径
j=0
for derName, subfolders, filenames in os.walk('/home/lifan/share/ML/img/'): #存放待移动文件的文件夹路径
    # print(derName)
    # print(subfolders)
    # print(filenames)
    for i in range(len(filenames)):
        if filenames[i].endswith('.jpg') or filenames[i].endswith('.png'):
            file_path=derName+'/'+filenames[i]
            newpath=new_path+'/'+filenames[i]
            shutil.copy(file_path,newpath)
            j+=1
            print(j)
            
#该脚本用于将多个子文件夹下的文件移动到指定文件夹下
