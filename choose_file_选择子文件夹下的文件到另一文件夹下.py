import shutil,os
new_path='/home/lifan/share/ML/video/image_2019-9-24/xml'
for derName, subfolders, filenames in os.walk('/home/lifan/share/ML/video/image_2019-9-24/label'):
    print(derName)
    print(subfolders)
    print(filenames)
    for i in range(len(filenames)):
        if filenames[i].endswith('.xml'):
            file_path=derName+'/'+filenames[i]
            newpath=new_path+'/'+filenames[i]
            shutil.copy(file_path,newpath)
