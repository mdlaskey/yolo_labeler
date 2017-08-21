import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import argparse
import yolo.config_bed_mac as cfg

from utils.timer import Timer
import IPython

#R = 0, B = 1, G = 2

BASE_COLOR = 2

def make_dim_same(b_img,dist):

    h,w,c = b_img.shape 

    dist = cv2.resize(dist,(w,h))

    return dist


def label_conversion(pose,dist):


    x = int(pose[0])/cfg.RESOLUTION
    y = int(pose[1])/cfg.RESOLUTION

  
    dist[y,x] = 1.0

    return dist


def viz_distribution(background,dist,pose=None):


    dist = label_conversion(pose,dist)

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
     
    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)/255

    #outImage_B = cv2.resize(outImage,(500,500))

    return outImage



   

