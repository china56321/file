# -*- coding:UTF-8 -*-
# xml_to_yolo_txt.py
import glob
import xml.etree.ElementTree as ET

#  类名
class_names = ['car', 'bus', 'person', 'bicycle', 'truck', 'motorbike']
#  转换一个xml文件为txt
def single_xml_to_txt(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	#  保存的txt文件路径
	# txt_file = xml_file.split('.')[0]+'.txt'
	txt_file = xml_file.split('.')[0]+'.'+xml_file.split('.')[1]+'.txt'
	with open(txt_file, 'w') as txt_file:
		for member in root.findall('object'):
			#filename = root.find('filename').text
			picture_width = int(root.find('size')[0].text)
			picture_height = int(root.find('size')[1].text)
			class_name = member[0].text
			#  类名对应的index
			class_num = class_names.index(class_name)
            
			box_x_min = int(member[4][0].text)  # 左上角横坐标 or [1][0]
			box_y_min = int(member[4][1].text)  # 左上角纵坐标 or [1][1]
			box_x_max = int(member[4][2].text)  # 右下角横坐标 or [1][2]
			box_y_max = int(member[4][3].text)  # 右下角纵坐标 or [1][3]
			# 转成相对位置和宽高
			x_center = (box_x_min + box_x_max) / (2.0 * picture_width)
			y_center = (box_y_min + box_y_max) / (2.0 * picture_height)
			width = (box_x_max - box_x_min) / (picture_width)
			height = (box_y_max - box_y_min) / ( picture_height)
			print(class_num, x_center, y_center, width, height)
			txt_file.write(str(class_num) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')

#  转换文件夹下的所有xml文件为txt
def dir_xml_to_txt(path):
    i=1
    for xml_file in glob.glob(path + '*.xml'):
        print("processing {}, {}".format(i, xml_file+'.xml'))
        single_xml_to_txt(xml_file)
        i += 1
        

def main(path):

    dir_xml_to_txt(path)

if __name__ == '__main__':
    #  xml文件路径
    # path = '/home/ambavm/make/YOLO-v5/datasets/score/images/Annotations/'
    path = '/home/ambavm/make/YOLO-v5/datasets/score/images/valxml/'
    main(path)

