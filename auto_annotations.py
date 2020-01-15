# -*- coding: utf-8 -*-
'''
本脚本基于refinedet模型检测样本，并保存XML标签.
'''
import cv2
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def ShowResults(img, image_file, results, labelmap, threshold=0.6, save_fig=False):
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    num_classes = len(labelmap.item) - 1
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    
    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue

        label = int(results[i, -1])
        name = get_labelname(labelmap, label)[0]
        a=[]
        a.append(name)
        print(a)
        print("/*/*/*/*")
        color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        print(xmin,ymin,xmax,ymax)
        print("************")
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        display_text = '%s: %.2f' % (name, score)
        ax.text(xmin, ymin, display_text, bbox={'facecolor':color, 'alpha':0.5})
        # plt.savefig(image_file[:-4] + '_dets.jpg',dpi=100,bbox_inches = 'tight')  #add_code
        # plt.savefig(image_file[:-4] + '_dets.jpg',dpi=500) 

        # open the crospronding txt file
        # write in xml file
        # xml_file = open((xml_name + '.xml'), 'w')
        # xml_file.write('<?xml version="1.0" ?>\n')
        # xml_file.write('<annotation>\n')
        # xml_file.write('    <folder>VOC2007</folder>\n')
        # xml_file.write('    <filename>' + xml_name + '.jpg'+ '</filename>\n')
        # xml_file.write('    <source>\n')
        # xml_file.write('        <database>The AUTO Database</database>\n')
        # xml_file.write('        <annotation>AUTO by zxl</annotation>\n')
        # xml_file.write('        <image>flickr</image>\n')
        # xml_file.write('    </source>\n')
        # xml_file.write('    <size>\n')
        # xml_file.write('        <width>' + str(width) + '</width>\n')
        # xml_file.write('        <height>' + str(height) + '</height>\n')
        # xml_file.write('        <depth>3</depth>\n')
        # xml_file.write('    </size>\n')
        # write the region of image on xml file
        for j in range(0,len(results.shape)):
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + str(name) + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
            j-=1
    xml_file.write('</annotation>')
    print("---------------done------------------------")
           
            # if j<=0:
            #     continue

    if save_fig:
        plt.savefig(image_file[:-4] + '_dets.jpg', bbox_inches="tight")
        print('Saved: ' + image_file[:-4] + '_dets.jpg')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_fig', action='store_true')
    args = parser.parse_args()

    # gpu preparation
    # if args.gpu_id >= 0:
    #     caffe.set_device(args.gpu_id)
    #     caffe.set_mode_gpu()

    # load labelmap
    labelmap_file = '/home/RefineDet/data/VOCdevkit/VOC2007/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    model_def = '/home/RefineDet/models/ResNet/coco/refinedet_resnet50_512x256/deploy.prototxt'
    model_weights = '/home/RefineDet/models/ResNet/coco/refinedet_resnet50_512x256/coco_refinedet_resnet50_512x256_iter_350000.caffemodel'



    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    if '320' in model_def:
        img_resize = 320
    else:
        img_resize = 512
    net.blobs['data'].reshape(1, 3, img_resize, img_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # im_names = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg','5.jpg','6.jpg','7.jpg','12.jpg','13.jpg','15.jpg',
    #             '17.jpg','19.jpg','21.jpg','22.jpg','24.jpg','25.jpg','61.jpg','30.jpg',
    #            '39.jpg','41.jpg','45.jpg','46.jpg','94.jpg','106.jpg','107.jpg','123.jpg','128.jpg','90.jpg','95.jpg','97.jpg','104.jpg','105.jpg','135.jpg']  
    # for im_name in im_names:
    #     # image_file = '/home/lifan/share/make/RefineDet/examples/images/coco/' + im_name
    #     image_file = '/home/ambavm/RefineDet/examples/images/' + im_name
    #     image = caffe.io.load_image(image_file)
    #     transformed_image = transformer.preprocess('data', image)
    #     net.blobs['data'].data[...] = transformed_image

    #     detections = net.forward()['detection_out']
    #     det_label = detections[0, 0, :, 1]
    #     det_conf = detections[0, 0, :, 2]
    #     det_xmin = detections[0, 0, :, 3] * image.shape[1]
    #     det_ymin = detections[0, 0, :, 4] * image.shape[0]
    #     det_xmax = detections[0, 0, :, 5] * image.shape[1]
    #     det_ymax = detections[0, 0, :, 6] * image.shape[0]
    #     result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
    
    im_names = os.listdir('/home/ambavm/RefineDet/examples/images/')
    for label_name in im_names:
        image_file = '/home/ambavm/RefineDet/examples/images/' + label_name
        image = caffe.io.load_image(image_file) #加载图片  
        img_size = image.shape
        print(img_size)
        height=img_size[0]
        width=img_size[1]
        print(height)
        print(width)

        transformed_image = transformer.preprocess('data', image) #执行上面设置的图片预处理操作，并将图片载入到blob中  
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * image.shape[1]
        det_ymin = detections[0, 0, :, 4] * image.shape[0]
        det_xmax = detections[0, 0, :, 5] * image.shape[1]
        det_ymax = detections[0, 0, :, 6] * image.shape[0]
        result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])

        xml_name=image_file.split(".")[0]
        xml_file = open((xml_name + '.xml'), 'w')
        xml_file.write('<?xml version="1.0" ?>\n')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + xml_name + '.jpg'+ '</filename>\n')
        xml_file.write('    <source>\n')
        xml_file.write('        <database>The AUTO Database</database>\n')
        xml_file.write('        <annotation>AUTO by zxl</annotation>\n')
        xml_file.write('        <image>flickr</image>\n')
        xml_file.write('    </source>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')
        # show result
        ShowResults(image, image_file, result, labelmap, 0.45, save_fig=args.save_fig)

    




























