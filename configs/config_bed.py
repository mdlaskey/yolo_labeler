import os

#
# path and dataset parameter
#



###############PARAMETERS TO SWEEP##########

FIXED_LAYERS = 33

#VARY {0, 4, 9}
# SS_DATA = 0
CONFIG_NAME = 'SS_0'




ROOT_DIR = '/media/autolab/1tb/data/'

NET_NAME = '08_28_01_37_11save.ckpt-30300'
DATA_PATH = ROOT_DIR + 'bed_rcnn/'

ROLLOUT_PATH = DATA_PATH+'rollouts/'
BC_HELD_OUT = DATA_PATH+'held_out_bc'
IMAGE_PATH = DATA_PATH+'images/'
LABEL_PATH = DATA_PATH+'labels/'

CACHE_PATH = DATA_PATH+'cache/'

OUTPUT_DIR = DATA_PATH +'output/'

TRAN_OUTPUT_DIR = DATA_PATH +'transition_output/' 
TRAN_STATS_DIR = TRAN_OUTPUT_DIR + 'stats/'
TRAIN_STATS_DIR_T = TRAN_OUTPUT_DIR + 'train_stats/'
TEST_STATS_DIR_T = TRAN_OUTPUT_DIR + 'test_stats/'


GRASP_OUTPUT_DIR = DATA_PATH + 'grasp_output/'
GRASP_STAT_DIR = GRASP_OUTPUT_DIR + 'stats/' 
TRAIN_STATS_DIR_G = GRASP_OUTPUT_DIR + 'train_stats/'
TEST_STATS_DIR_G = GRASP_OUTPUT_DIR + 'test_stats/'

WEIGHTS_DIR = DATA_PATH + 'weights/'

PRE_TRAINED_DIR = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/'

ROLLOUT_PATH = DATA_PATH+'rollouts/'

WEIGHTS_FILE = None


# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['yes','no']

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
T_IMAGE_SIZE_H = 480
T_IMAGE_SIZE_W = 640

IMAGE_SIZE = 448
CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = True



RESOLUTION = 10


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.1
LEARNING_RATE_C = 0.1


DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 45

#MAX_ITER = 200
MAX_ITER = 2000#30000

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
