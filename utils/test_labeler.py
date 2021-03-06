import os,sys
sys.path.append()
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy
import glob
import yolo.config as cfg
import cPickle as pickle
import IPython

class TestLabeler(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')

        self.cache_path = cfg.CACHE_PATH
        self.image_path = cfg.IMAGE_PATH
        self.label_path = cfg.LABEL_PATH

        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
      



    def check_label(self,frame):

        label_path = cfg.LABEL_PATH+frame
        label_data = pickle.load(open(label,'r'))

        for objs in label_data['objects']:
            box_ind = objs['box_index']
            class_label = objs['num_class_label']
           
            print "CLASS LABEL"
            print class_label

            print "BOX INDEX"
            print box_ind


    def check_frame(self,frame):
        image_path = cfg.IMAGE_PATH+frame
        image = cv2.imread(imname)

        cv2.imshow(image)
        cv2.waitKey(0)



    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)

        image = cv2.resize(image, (self.image_size, self.image_size))
        # cv2.imshow('debug',image)
        # cv2.waitKey(30)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

  
    def load_bbox_annotation(self, label):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
 
        label_data = pickle.load(open(label,'r'))

        num_objs = label_data['num_labels']

        label = np.zeros((self.cell_size, self.cell_size, 5+cfg.NUM_LABELS))

        for objs in label_data['objects']:
            box_ind = objs['box_index']
            class_label = objs['num_class_label']
            x_ind = int(box_ind[0] * self.cell_size / self.image_size)
            y_ind = int(box_ind[1] * self.cell_size / self.image_size)

            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = box_ind
            label[y_ind, x_ind, 5 + class_label] = 1


        return label, num_objs




if __name__ == '__main__':
    tl = TestLabeler()

    frame = 'frame_1.png'

    tl.check_label(frame)

    tl.check_frame(frame)
