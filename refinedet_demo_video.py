'''
In this example, we will load a RefineDet model and use it to detect objects.
'''
from PIL import Image
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

# def ShowResults(img, image_file, results, labelmap, threshold=0.6, save_fig=False):
def ShowResults(img, results, labelmap, threshold=0.6, save_fig=False):
    # plt.clf()
    # plt.imshow(img)
    # plt.axis('off')
    # ax = plt.gca()

    num_classes = len(labelmap.item) - 1
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue

        label = int(results[i, -1])
        name = get_labelname(labelmap, label)[0]
        color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        # ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        # plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1)
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 1)
        display_text = '%s: %.2f' % (name, score)
        # ax.text(xmin, ymin, display_text, bbox={'facecolor':color, 'alpha':0.5})
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(image, display_text, (xmin,ymin-3), font, 0.6, color, 1)
        cv2.imshow('image',image)
        # plt.savefig(image_file[:-4] + '_dets.jpg',dpi=100,bbox_inches = 'tight')  #add_code
        # plt.savefig(image_file[:-4] + '_dets.jpg',dpi=500) 
    # if save_fig:
    #     plt.savefig(image_file[:-4] + '_dets.jpg', bbox_inches="tight")
    #     print('Saved: ' + image_file[:-4] + '_dets.jpg')
    plt.show()
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_fig', action='store_true')
    args = parser.parse_args()

    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    # load labelmap
    labelmap_file = '/home/share/make/RefineDet/data/VOCdevkit/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    model_def = '/home/share/make/RefineDet/models/ResNet/coco/refinedet_resnet50_512x256/deploy.prototxt'
    # model_weights ='/home/share/make/RefineDet/models/ResNet/coco/refinedet_resnet50_512x256/coco_refinedet_resnet50_512x256_iter_1140000.caffemodel'
    model_weights ='/home/share/make/RefineDet/models/ResNet/coco/refinedet_resnet50_512x256/coco_refinedet_resnet50_512x256_iter_1368000.caffemodel'



    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    img_resize_width = 512
    img_resize_height = 256
    net.blobs['data'].reshape(1, 3, img_resize_width, img_resize_height)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    

    input_dir = '/home/share/make/RefineDet/test/test/cell/2.MP4'
  
    
    cap = cv2.VideoCapture(input_dir)  

    while cap.isOpened():
        ret, image = cap.read()
        if ret == True:
        # cv2.imshow('image', frame)  
		# for im_name in im_names:
        # image_file = '/home/share/make/RefineDet/examples/images/coco/' + im_name
        # image_file = '/home/RefineDet/examples/images/' + im_name
          image = image/ 255.0
          transformed_image = transformer.preprocess('data', image)
          net.blobs['data'].data[...] = transformed_image

          detections = net.forward()['detection_out']
          det_label = detections[0, 0, :, 1]
          det_conf = detections[0, 0, :, 2]
          det_xmin = detections[0, 0, :, 3] * image.shape[1]
          det_ymin = detections[0, 0, :, 4] * image.shape[0]
          det_xmax = detections[0, 0, :, 5] * image.shape[1]
          det_ymax = detections[0, 0, :, 6] * image.shape[0]
          result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
        
          ShowResults(image, result, labelmap, 0.45, save_fig=args.save_fig)
          # ShowResults(image, image_file, result, labelmap, 0.45, save_fig=args.save_fig)
        k = cv2.waitKey(20)  
        #q exit
        if (k & 0xff == ord('q')):  
           break  

    cap.release()  
    cv2.destroyAllWindows()






























