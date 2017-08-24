import numpy as np
import tensorflow as tf
#import yolo.config_card as cfg
import configs.config_bed as cfg
import IPython

slim = tf.contrib.slim


class SNet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.dist_size_w = cfg.T_IMAGE_SIZE_W/cfg.RESOLUTION
        self.dist_size_h = cfg.T_IMAGE_SIZE_H/cfg.RESOLUTION
        self.output_size = 2
       
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA


        self.images = tf.placeholder(tf.float32, [None, cfg.FILTER_SIZE, cfg.FILTER_SIZE, cfg.NUM_FILTERS], name='images')
        self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        if is_training:
            
            self.labels = tf.placeholder(tf.float32, [None, 2])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=1.0,
                      is_training=True,
                      scope='yolo'):
       
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
             
                net = tf.pad(images, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob,
                                   is_training=is_training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs,
                                           activation_fn=None, scope='fc_36')
                net = tf.nn.softmax(net)

                
        return net



    def loss_layer(self, predicts, classes, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts, [self.batch_size,2])
            
            # class_loss

            class_delta = (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1]), name='class_loss') 

            tf.losses.add_loss(class_loss)
          

            tf.summary.scalar('class_loss', class_loss)
          


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op
