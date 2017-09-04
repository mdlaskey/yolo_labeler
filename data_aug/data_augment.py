import cv2
import cPickle as pickle
import IPython
import numpy as np
import configs.config_bed as cfg
from augment_lighting import get_lighting
import copy
HALF_LENGTH = 15

RADIUS = 10

THICKNESS = 1
C_THICKNESS = 1

COLOR = (0,0,255) #RGB


def flip_data_vertical(img,label):

	h,w,channel = img.shape

	v_img = cv2.flip(img,1)

	label[0] = w-label[0]

	return {'c_img': v_img, 'pose': label}


def flip_data_horizontal(img,label):

	h,w,channel = img.shape

	h_img = cv2.flip(img,0)

	label[1] = h-label[1]

	return {'c_img': h_img, 'pose': label}


	


def augment_data(data):

	augmented_data = []

	img = data['c_img']
	label = data['pose']

	if label == None: 
		label = [0,0]
	
	light_imgs = get_lighting(img)

	for l_imgs in light_imgs:

		p_n = {'c_img': l_imgs, 'pose': label}
		
		#p_h = flip_data_horizontal(l_imgs,np.copy(label))

		p_v = flip_data_vertical(l_imgs,np.copy(label))

		augmented_data.append(p_n)
		#augmented_data.append(p_h)
		augmented_data.append(p_v)


	return augmented_data




	





if __name__ == "__main__":


	dp = DrawPrediction()

	path = cfg.ROLLOUT_PATH+'rollout_0/rollout.p'
	data = pickle.load(open(path,'rb'))

	grasp_point = data[0]

	box = grasp_point['label']['objects'][0]['box']

	x = int((box[0] + box[2])/2.0) 

	y = int((box[1]+ box[3])/2.0)

	pose = [x,y]



	c_img = grasp_point['c_img']
	


	image = dp.draw_prediction(c_img,pose)

	cv2.imshow('debug', image)
	cv2.waitKey(0)

	print "RESULT ", sc.check_success(wl)