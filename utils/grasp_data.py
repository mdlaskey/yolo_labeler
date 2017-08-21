import os
import xml.etree.ElementTree as ET
import numpy as np
from numpy.random import random
import cv2
import cPickle
import copy
import glob
from utils.augment_lighting import get_lighting
from debug.visualizer_training import VTraining
import yolo.config_card as cfg
from yolo.yolo_conv_features import YOLO_CONV
import cPickle as pickle
import IPython

class bbox_data(object):
    def __init__(self, phase, rebuild=False):

        self.rollout_path = cfg.ROLLOUT_PATH
       

        self.image_size = cfg.IMAGE_SIZE
        self.dist_size_w = cfg.T_IMAGE_SIZE_W/cfg.RESOLUTION
        self.dist_size_h = cfg.T_IMAGE_SIZE_H/cfg.RESOLUTION
        self.output_size = self.dist_size_h*self.dist_size_w
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.noise = cfg.LIGHTING_NOISE 
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.t_cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.test_labels = None

        self.vtraining = VTraining()

        self.yc = YOLO_CONV()
        self.yc.load_network()

        self.recent_batch = []
        self.prepare()

    def get(self, noise=False):
        images = np.zeros((self.batch_size, cfg.FILTER_SIZE, cfg.FILTER_SIZE, cfg.NUM_FILTERS))
        labels = np.zeros((self.batch_size, self.dist_size_h,self.dist_size_w))
        count = 0
        self.recent_batch = []
        while count < self.batch_size:

            
            images[count, :, :, :] = self.train_labels[self.cursor]['features']
            labels[count, :, :] = self.train_labels[self.cursor]['label']

            self.recent_batch.append(self.train_labels[self.cursor])

            count += 1
            if(count == self.batch_size):
                break

            self.cursor += 1
            if self.cursor >= len(self.train_labels):
                np.random.shuffle(self.train_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def get_test(self):
        images = np.zeros((self.batch_size, cfg.FILTER_SIZE, cfg.FILTER_SIZE, cfg.NUM_FILTERS))
        labels = np.zeros((self.batch_size, self.dist_size_h,self.dist_size_w))

        count = 0
        while count < self.batch_size:
           
            images[count, :, :, :] = self.test_labels[self.t_cursor]['features']
            labels[count, :, :] = self.test_labels[self.t_cursor]['label']
            count += 1
            self.t_cursor += 1
            if self.t_cursor >= len(self.test_labels):
                np.random.shuffle(self.test_labels)
                self.t_cursor = 0
                self.epoch += 1
        return images, labels

    
    def prep_image(self, image):
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0

        return image


    def load_rollouts(self):
       
        self.train_labels = []
        self.test_labels = []
        labels = glob.glob(os.path.join(self.rollout_path, '*_*'))

        count = 0
      
        for rollout_p in rollouts:
          
            rollout = pickle.load(open(rollout_p+'/rollout.p'))

            if(random() > 0.2):
                training = True
            else: 
                training = False

            for data in rollout:

                if(data['type'] = 'grasp'):
                    
                    data_a = aug_data(data)
                    
                    for datum_a in data_a:
                        im_r = self.prep_image(datum_a['c_img'])
                        features = self.yc.extract_conv_features(im_r)

                        label = self.compute_label(datum_a['pose'])

                        if training:
                            train_labels.append({'rname': rollout_p, 'label': label_num, 'features':features})
                        else:
                            test_labels.append({'rname': rollout_p, 'label': label_num, 'features':features})
  
        return gt_labels

 


    def compute_label(self, label):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        label = np.zeros((self.batch_size, self.dist_size_h,self.dist_size_w))

        ind_x = int(pose[0])/cfg.RESOLUTION
        ind_y = int(pose[1])/cfg.RESOLUTION

        label[ind_y,ind_x] = 1.0

        return label


     
        

