import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import argparse
import configs.config_bed as cfg

from utils.timer import Timer
import IPython

class VTraining(object):

    def __init__(self):
        
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

       

    def draw_result(self, img, results):
        img_h, img_w, _ = img.shape
        for i in range(len(results)):

            x = int(results[i]['box'][0])
            y = int(results[i]['box'][1])
            w = int(results[i]['box'][2] / 2)
            h = int(results[i]['box'][3] / 2)

            #IPython.embed()
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, results[i]['class'] + ' : %.2f' % results[i]['prob'], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)

 
    def detect(self,sess,net, inputs,image):
        img_h, img_w, _ = image.shape
        net_output = sess.run(net.logits,
                                   feed_dict={net.images: inputs})
        #IPython.embed()
       
        for i in range(net_output.shape[0]):
            results = self.interpret_output(net_output[i])
            
        print "results ", results
        for i in range(len(results)):
            
            results[i]['box'][0] *= (1.0 * img_w / self.image_size)
            results[i]['box'][1] *= (1.0 * img_h / self.image_size)
            results[i]['box'][2] *= (1.0 * img_w / self.image_size)
            results[i]['box'][3] *= (1.0 * img_h / self.image_size)

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])
        
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
            0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        results = []
        for i in range(len(boxes_filtered)):
            result = {}
            result['class'] = self.classes[classes_num_filtered[i]]
            result['box'] = [boxes_filtered[i][0], boxes_filtered[
                          i][1], boxes_filtered[i][2], boxes_filtered[i][3]]
            result['prob'] = probs_filtered[i]

            results.append(result)
           

        return results

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


    def label_detector(self,sess,net,labels, wait=0):
        

        for label in labels:
            imname = label['imname']
            features = label['features']

            image = cv2.imread(imname)

           
            result = self.detect(sess,net,features,image)
            

            #print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

            self.draw_result(image, result)
            # cv2.imshow('Image', image)
            # cv2.waitKey(wait)
            framename = os.path.split(imname)[-1].split('.')[0]
            cv2.imwrite("debug/training_viz/" + framename + ".png", image)



       




if __name__ == '__main__':
    main()
