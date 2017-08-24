import os
import xml.etree.ElementTree as ET
import numpy as np
from numpy.random import random
import cv2
import cPickle
import copy
import glob
from data_aug.data_augment import augment_data
from debug.visualizer_training import VTraining
from visualizer.alpha_blend import viz_distribution,plot_prediction
import configs.config_bed as cfg
from yolo.yolo_conv_features import YOLO_CONV
import cPickle as pickle
import IPython

class success_data(object):
    def __init__(self, phase, rebuild=False):

        self.rollout_path = cfg.ROLLOUT_PATH
        self.batch_size = cfg.BATCH_SIZE
       
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
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
        self.load_rollouts()

    def get(self, noise=False):
        images = np.zeros((self.batch_size, cfg.FILTER_SIZE, cfg.FILTER_SIZE, cfg.NUM_FILTERS))
        labels = np.zeros((self.batch_size, 2))
        count = 0
        self.recent_batch = []
        while count < self.batch_size:

            
            images[count, :, :, :] = self.train_labels[self.cursor]['features']
           
            labels[count, :] = self.train_labels[self.cursor]['label']

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
        labels = np.zeros((self.batch_size, 2))

        count = 0
        while count < self.batch_size:
           
            images[count, :, :, :] = self.test_labels[self.t_cursor]['features']
            labels[count, :] = self.test_labels[self.t_cursor]['label']
            count += 1
            self.t_cursor += 1
            if self.t_cursor >= len(self.test_labels):
                np.random.shuffle(self.test_labels)
                self.t_cursor = 0
                self.epoch += 1
        return images, labels


    def viz_debug(self,sess,net):
        return




    
    def prep_image(self, image):
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0

        return image


    def load_rollouts(self):
       
        self.train_labels = []
        self.test_labels = []
        rollouts = glob.glob(os.path.join(self.rollout_path, '*_*'))

        count = 0
        
        #for rollout_p in rollouts:
        rollout_p = rollouts[0]  
        rollout = pickle.load(open(rollout_p+'/rollout.p'))

        # if(random() > 0.2):
        #     training = True
        # else: 
        #     training = False
        training = True
        print rollout_p
        print len(rollout)
        for data in rollout:

            if(data['type'] == 'success'):
                
                data_a = augment_data(data)
                
                for datum_a in data_a:
                    im_r = self.prep_image(datum_a['c_img'])
                    features = self.yc.extract_conv_features(im_r)

                    label = self.compute_label(data['class'])

                    if training:
                        self.train_labels.append({'c_img': datum_a['c_img'], 'label': label, 'features':features})
                    else:
                        self.test_labels.append({'c_img': datum_a['c_img'], 'label': label, 'features':features})
  
        return 

 


    def compute_label(self, clss):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        label = np.zeros((2))
        label[clss] = 0.9
       
        return label


     
        

