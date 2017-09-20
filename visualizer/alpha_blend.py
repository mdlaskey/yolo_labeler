import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import argparse
import configs.config_bed as cfg

from utils.timer import Timer
import IPython
from data_aug.draw_cross_hair import DrawPrediction
#R = 0, B = 1, G = 2

BASE_COLOR = 2

DP = DrawPrediction()

def make_dim_same(b_img,dist):

    h,w,c = b_img.shape 

    dist = cv2.resize(dist,(w,h))

    return dist


def plot_label_conversion(outImage,dist):

    x,y = np.unravel_index(dist.argmax(),dist.shape)

    pose = [x,y]
    img = DP.draw_prediction(outImage,pose)

    return img


def viz_distribution(background,dist):

    dist = make_dim_same(background,dist)

    foreground = np.zeros(background.shape)
    dist_c = np.zeros(background.shape)
    foreground[:,:,BASE_COLOR] = 255.0

    dist_c[:,:,0] = dist
    dist_c[:,:,1] = dist
    dist_c[:,:,2] = dist



    foreground = foreground.astype(float)
    background = background.astype(float)
 
    
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(dist_c, foreground)
     
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - dist_c, background)
    #IPython.embed()
    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)/255

    outImage = plot_label_conversion(outImage,dist)

    #outImage_B = cv2.resize(outImage,(500,500))

    return outImage



def plot_prediction(outImage,pose):

    print pose
  
    x = cfg.T_IMAGE_SIZE_W*(pose[0,0]+0.5)
    y = cfg.T_IMAGE_SIZE_H*(pose[0,1]+0.5)

    pose = [x,y]
    img = DP.draw_prediction(outImage,pose)

    return img

   

