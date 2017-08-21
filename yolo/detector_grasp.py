import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import argparse
import yolo.config_bed_mac as cfg
from yolo.grasp_heat_net import GHNet
from yolo.yolo_conv_features import YOLO_CONV
from utils.timer import Timer
import IPython
import sys, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
slim = tf.contrib.slim

class Detector(object):

    def __init__(self):
        
        

        self.yc = YOLO_CONV(is_training = False)
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)

        self.image_size = cfg.IMAGE_SIZE
        self.dist_size_w = cfg.T_IMAGE_SIZE_W/cfg.RESOLUTION
        self.dist_size_h = cfg.T_IMAGE_SIZE_H/cfg.RESOLUTION
        self.output_size = self.dist_size_h*self.dist_size_w

        self.yc.load_network()
        self.count = 0


        #self.all_data = self.precompute_features(images)

        self.load_trained_net()

        #self.images_detectors()

       

    def load_trained_net(self):
        self.sess = tf.Session()
        self.net = GHNet(is_training = False)
        self.sess.run(tf.global_variables_initializer())
        
        # trained_model_file = cfg.OUTPUT_DIR+ cfg.NET_NAME
        # print 'Restoring weights from: ' + trained_model_file
        # self.variable_to_restore = slim.get_variables_to_restore()
        # count = 0
        # for var in self.variable_to_restore:
        #     print str(count) + " "+ var.name
        #     count += 1
        
        # self.variables_to_restore = self.variable_to_restore[40:]
        # self.saver_f = tf.train.Saver(self.variables_to_restore, max_to_keep=None)

        # self.saver_f.restore(self.sess, trained_model_file)

        
    def precompute_features(self,images):

        all_data = []
        for image in images:

            features = self.yc.extract_conv_features(image)

            data = {}
            data['image'] = image
            data['features'] = features
            all_data.append(data)

        return all_data


    def detect(self,inputs,image):
        img_h, img_w, _ = image.shape
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        
        results = self.interpret_output(net_output)
            
        return results

    def detect_from_cvmat(self, inputs):
        print inputs
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        #IPython.embed()
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        img_dist = np.reshape(output,[self.dist_size_h,self.dist_size_w])

        return img_dist


    def numpy_detector(self,image):
       

        features = self.yc.extract_conv_features(image)
   
        result = self.detect(features,image)
       

        return result

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        features = self.yc.extract_conv_features(image)

        detect_timer.tic()
        result = self.detect(features,image)
        detect_timer.toc()

        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        self.draw_result(image, result)
        # cv2.imshow('Image', image)
        # cv2.waitKey(wait)
        framename = 'frame_'+str(self.count)
        self.count += 1
        cv2.imwrite("heldOutTests2/" + framename + ".png", image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

   
    weight_file = "/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/output07_20_09_37_29save.ckpt-15000"
   
   
    images = []
    c = 0

    detector = Detector()

    rollout  = pickle.load(open(cfg.ROLLOUT_PATH+'rollout.p','rb'))

    img = rollout[0]['c_img']
    for imname in imageList:

        detector.image_detector(imname)
    

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from held out image file

    


if __name__ == '__main__':
    main()
