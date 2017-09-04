import cv2
import cPickle as pickle
import IPython
import numpy as np
import configs.config_bed as cfg

HALF_LENGTH = 15

RADIUS = 10

THICKNESS = 1
C_THICKNESS = 1

COLOR = (0,0,255) #RGB


class DrawPrediction:


	def make_cross_hair(self,image,p):
	
		cv2.circle(image,p,RADIUS,COLOR,C_THICKNESS)

		p1_h = (p[0] - HALF_LENGTH, p[1])
		p2_h = (p[0] + HALF_LENGTH, p[1])
		cv2.line(image,p1_h,p2_h,COLOR,THICKNESS)

		p1_v = (p[0], p[1] - HALF_LENGTH)
		p2_v = (p[0],p[1] + HALF_LENGTH)
		cv2.line(image,p1_v,p2_v,COLOR,THICKNESS)

		#IPython.embed()
		return image


	def draw_prediction(self,image,pose):

		
		x,y = pose
		pose = [int(x),int(y)]
		pose = tuple(pose)
		image = self.make_cross_hair(image,pose)

		return image



if __name__ == "__main__":


	dp = DrawPrediction()

	path = cfg.ROLLOUT_PATH+'rollout_0/rollout.p'
	data = pickle.load(open(path,'rb'))

	grasp_point = data[0]

	box = grasp_point['label']['objects'][0]['box']

	x = int((box[0] + box[2])/2.0) 

	y = int((box[1]+ box[3])/2.0)

	pose = (x,y)



	c_img = grasp_point['c_img']
	


	image = dp.draw_prediction(c_img,pose)

	cv2.imshow('debug', image)
	cv2.waitKey(0)

	print "RESULT ", sc.check_success(wl)









