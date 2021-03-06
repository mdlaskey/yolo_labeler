import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import argparse
import yolo.config as cfg
from yolo.yolo_net_fast import YOLONet
from utils.timer import Timer
import IPython

class Detector(object):

    def __init__(self):
        self.net = YOLONet(False)
        self.weights_file = cfg.OUTPUT_DIR+weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.net.load_network()

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

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        #inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        results = self.detect_from_cvmat(inputs)[0]
        
        for i in range(len(results)):
            results[i]['box'][0] *= (1.0 * img_w / self.image_size)
            results[i]['box'][1] *= (1.0 * img_h / self.image_size)
            results[i]['box'][2] *= (1.0 * img_w / self.image_size)
            results[i]['box'][3] *= (1.0 * img_h / self.image_size)

        return results

    def detect_from_cvmat(self, inputs):
        
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        #IPython.embed()
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

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

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(0)
            IPython.embed()

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        self.draw_result(image, result)
        # cv2.imshow('Image', image)
        # cv2.waitKey(wait)
        framename = os.path.split(imname)[-1].split('.')[0]
        cv2.imwrite("heldOutTests/" + framename + ".png", image)


    def numpy_detector(self,image,wait=0):
        detect_timer = Timer()
        

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Camera',image)
        cv2.waitKey(30)
        return result
       


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    weight_file = "/media/autolab/1tb/data/hsr_clutter_rcnn/output/07_14_15_27_17save.ckpt-12000"
    #weight_file = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/save.ckpt-100'#os.path.join(args.data_dir, args.weight_dir, args.weights)
    #weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)

    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from held out image file

    imageList = glob.glob(os.path.join(cfg.IMAGE_PATH, '*.png'))
    labelListOrig = glob.glob(os.path.join(cfg.LABEL_PATH, '*.p'))
    labelList = [os.path.split(label)[-1].split('.')[0] for label in labelListOrig]
    #remove labeled images
    imageList = [img for img in imageList if os.path.split(img)[-1].split('.')[0] not in labelList]

    #imname = cfg.IMAGE_PATH + 'frame_1000.png'
    #imname = 'test/person.jpg' 
    for imname in imageList[:100]:
        detector.image_detector(imname)


if __name__ == '__main__':
    main()
