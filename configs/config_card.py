import os

#
# path and dataset parameter
#

ROOT_DIR = '/media/autolab/1tb/data/'

NET_NAME = '07_31_00_09_46save.ckpt-30300'
DATA_PATH = ROOT_DIR + 'card_rcnn/'

IMAGE_PATH = DATA_PATH+'images/'
LABEL_PATH = DATA_PATH+'labels/'

CACHE_PATH = DATA_PATH+'cache/'

OUTPUT_DIR = DATA_PATH +'output/'

TRAIN_STATS_DIR = OUTPUT_DIR + 'train_stats/'
TEST_STATS_DIR = OUTPUT_DIR + 'test_stats/'

WEIGHTS_DIR = DATA_PATH + 'weights/'

PRE_TRAINED_DIR = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/'

WEIGHTS_FILE = None


# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['up','down']

# #CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
# 			'train', 'tvmonitor']

NUM_LABELS = len(CLASSES)

FLIPPED = False
LIGHTING_NOISE = True




#
# model parameter
#

#IMAGE_SIZE = 250
IMAGE_SIZE = 448
CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = True

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 45

MAX_ITER = 30000

SUMMARY_ITER = 10
TEST_ITER = 20
SAVE_ITER = 1000

VIZ_DEBUG_ITER = 2000
#
# test parameter
#

#THRESHOLD = 0.0008
PICK_THRESHOLD = 0.4
THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
#IOU_THRESHOLD = 0.0001

#FAST PARAMS
FILTER_SIZE = 14
NUM_FILTERS = 1024
