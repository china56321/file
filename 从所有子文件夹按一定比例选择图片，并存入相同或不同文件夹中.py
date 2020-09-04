# -*- coding:UTF-8 -*-
'''
该脚本主要从所有子文件夹中按一定比例选择图片，并存入相同文件名称或重命名的文件夹中

--1>子文件夹重命名
--2>子文件夹名称相同

'''

#****************文件夹名称重命名－－－1****************************************
# # -*- coding:UTF-8 -*-
# import shutil,os,random
# new_path='/home/ecl/make/test/test/' 

# train_path='/home/ecl/make/test/train/'

# subfolders=os.listdir(train_path)

# i=0000
# for subfolder in subfolders:
#     test_path=new_path+str(i).zfill(4)
#     os.makedirs(test_path)
#     image_names=os.listdir(train_path + subfolder)
#     rate=0.3
#     img_num=int(len(image_names)*rate)   
#     sample = random.sample(image_names, img_num)
#     for name in sample:
#         shutil.copy(train_path + subfolder +'/'+ name,test_path)
#     i+=2

#*********************************************************************

        
#**************以相同文件夹的名称命名－－－2*****************************
# -*- coding:UTF-8 -*-
import shutil,os,random
new_path='/home/ecl/make/test/test/' 

train_path='/home/ecl/make/test/train/'

subfolders=os.listdir(train_path)

i=0000
for subfolder in subfolders:

    test_path=new_path+subfolder+'/'
    os.makedirs(new_path+subfolder)
    image_names=os.listdir(train_path + subfolder)
    rate=0.3
    img_num=int(len(image_names)*rate) 
    print(len(train_path + subfolder))
    sample = random.sample(image_names, img_num)
    for name in sample:
        shutil.copy(train_path + subfolder +'/'+ name,test_path)
    i+=2


#********************************************************
            