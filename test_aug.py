
import cPickle as pickle
import IPython
import numpy as np
import configs.config_bed as cfg
from data_aug.draw_cross_hair import DrawPrediction

from data_aug.data_augment import augment_data
import cv2


dp = DrawPrediction()

path = cfg.ROLLOUT_PATH+'rollout_0/rollout.p'
data = pickle.load(open(path,'rb'))

grasp_point = data[0]

box = grasp_point['label']['objects'][0]['box']

x = int((box[0] + box[2])/2.0) 

y = int((box[1]+ box[3])/2.0)

pose = [x,y]

c_img = grasp_point['c_img']

augmented_data = augment_data(c_img,pose)

count = 0

for img_a in augmented_data:

	img, label = img_a
	image = dp.draw_prediction(img,label)

	cv2.imwrite('debug/img_'+str(count)+'.png', image)
	count += 1




