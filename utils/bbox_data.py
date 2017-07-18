import os
import xml.etree.ElementTree as ET
import numpy as np
from numpy.random import random
import cv2
import cPickle
import copy
import glob
import yolo.config as cfg
import cPickle as pickle
import IPython

class bbox_data(object):
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
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.t_cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.test_labels = None
        self.prepare()

    def get(self, noise=False):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 5+cfg.NUM_LABELS))
        count = 0

        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped, noise=noise)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def get_test(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 5+cfg.NUM_LABELS))
        count = 0
        while count < self.batch_size:
            imname = self.test_labels[self.t_cursor]['imname']
            flipped = self.test_labels[self.t_cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.test_labels[self.t_cursor]['label']
            count += 1
            self.t_cursor += 1
            if self.t_cursor >= len(self.test_labels):
                np.random.shuffle(self.test_labels)
                self.t_cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False,noise=False):
        image = cv2.imread(imname)

        image = cv2.resize(image, (self.image_size, self.image_size))
        # cv2.imshow('debug',image)
        # cv2.waitKey(30)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        #with noise, no longer guaranteed to be between 0 and 255
        if noise:
            image = self.add_gnoise(image)

        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
    
        return image

    #from lighting tool
    def add_gnoise(self, img):
        row,col,ch = img.shape
        mean = 0
        var = 0.1
        sigma = 10
        gauss = np.random.normal(mean, sigma, (row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        return img + gauss

    def prepare(self):
        gt_labels = self.load_labels()
        np.random.shuffle(gt_labels)
        train_labels = []
        test_labels = []
        for datapoint in gt_labels:
            if (random() > 0.2):
                train_labels.append(datapoint)
            else:
                test_labels.append(datapoint)

        self.gt_labels = train_labels
        self.test_labels = test_labels
        return gt_labels

    def load_labels(self):
       

        print('Processing gt_labels from: ' + self.data_path)

        gt_labels = []
        labels = glob.glob(os.path.join(self.label_path, '*.p'))
        print(labels)
        for label in labels:
            print label
            imname = self.image_path + 'frame_'+ label[54:-2]+'.png'
            label_num, num = self.load_bbox_annotation(label,imname)
            if num == 0:
                continue

            #TODO: CHANGE THIS TO BE INDPENDENT OF DIRECTORY PATH
            
            gt_labels.append({'imname': imname, 'label': label_num, 'flipped': False})
  
        return gt_labels

 


    def load_bbox_annotation(self, label,img_path):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        imname = os.path.join(img_path)
        im = cv2.imread(imname)

        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label_data = pickle.load(open(label,'r'))

        num_objs = label_data['num_labels']

        label = np.zeros((self.cell_size, self.cell_size, 5+cfg.NUM_LABELS))

        for objs in label_data['objects']:
            box_ind = objs['box_index']
            class_label = objs['num_class_label']


            x1 = max(min((float(box_ind[0]) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(box_ind[1]) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(box_ind[2]) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(box_ind[3]) - 1) * h_ratio, self.image_size - 1), 0)

            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            print "CLASS ", class_label
            print boxes
            x_ind = int(box_ind[0] * self.cell_size / self.image_size)
            y_ind = int(box_ind[1] * self.cell_size / self.image_size)

            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + class_label] = 1

        return label, num_objs

