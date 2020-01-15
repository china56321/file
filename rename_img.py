# -*- coding:utf8 -*-
#!/usr/bin/python2.7
import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self):
        self.path = '/home/share/ML/160115000'  #路径 

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 24258 #图片名称起始值

        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                str1=str(i)
                dst = os.path.join(os.path.abspath(self.path), str1.zfill(6) + '.jpg') #图片名称为5位数，即第一张图片为000000，第二张图片为000001，以此类推.......
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
       # print 'total %d to rename & converted %d jpgs' % (total_num, i)

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
