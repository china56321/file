# -*- coding:UTF-8 -*-

import os

def mkd():
	k = 0000 # 两位数,不足向前取零
	path = "/home/lifan/share/ML/" #文件存哪里
	for i in range(1,8): #创建一个文件并循环20-1次
		file_name = str(k) #给文件命名 路径+ 文件标号（以标号为名字）
		os.makedirs(file_name) #创建文件
		print(file_name + "创建成功")
		k=k+1
		i = i+1

mkd() #函数调用
