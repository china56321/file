# -*- coding: utf-8 -*-

# author by LYS 2017/5/24
# for Deep Learning course
'''
python随机选取10000张图片并复制到另一个文件夹中
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil

def copyFile(fileDir):
    # 1
	pathDir = os.listdir(fileDir)
 
    # 2
	sample = random.sample(pathDir, 5000)
	print (sample)
	
    # 3
        i=0
	for name in sample:
		#shutil.copyfile(fileDir+name, tarDir+name)
                 shutil.move(fileDir+name, tarDir+name)
                 i+=1
                 print(i)
if __name__ == '__main__':

	fileDir = "/home/ambavm/BDD/train-img/"  
	tarDir = '/home/ambavm/BDD/choose_test_img/'
	copyFile(fileDir)




























