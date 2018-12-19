from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


class MobileNet(object):
    """docstring for MobileNet"""
    def __init__(self, inputs, num_classes, width_multiplier, is_training=True, scope="MobileNet"):
        self.inputs = inputs
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        with tf.variable_scope(scope) as sc:
            with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                activation=None,
                                weights_initializer=slim.initializers.xavier_initializer(),
                                biases_initializer=slim.init_ops.zeros_initializer()):
                with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=tf.nn.relu,
                                fused=True):
                    self._build_net()

    def _depthwise_separable_conv(self, inputs, num_output_channels, width_multiplier, sc, downsample=False):
        num_output_channels = round(num_output_channels * width_multiplier)
        _stride = 2 if downsample else 1

        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=width_multiplier,
                                                      kernel_size=[3, 3],
                                                      scope=sc + "/dw_conv")
        bn = slim.batch_norm(depthwise_conv, scope=sc + "/dw_bn")
        pointwise_conv = slim.convolution2d(bn, num_output_channels, kernel_size=[1, 1], scope=sc + '/pw_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_bn')
        return bn

    def _build_net(self):
        net = slim.convolution2d(self.inputs, round(32 * self.width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
        net = slim.batch_norm(net, scope='conv_1/bn')
        net = self._depthwise_separable_conv(net, 64, self.width_multiplier, sc='conv_ds_2')
        net = self._depthwise_separable_conv(net, 128, self.width_multiplier, downsample=True, sc='conv_ds_3')
        net = self._depthwise_separable_conv(net, 128, self.width_multiplier, sc='conv_ds_4')
        net = self._depthwise_separable_conv(net, 256, self.width_multiplier, downsample=True, sc='conv_ds_5')
        net = self._depthwise_separable_conv(net, 256, self.width_multiplier, sc='conv_ds_6')
        net = self._depthwise_separable_conv(net, 512, self.width_multiplier, downsample=True, sc='conv_ds_7')

        net = self._depthwise_separable_conv(net, 512, self.width_multiplier, sc='conv_ds_8')
        net = self._depthwise_separable_conv(net, 512, self.width_multiplier, sc='conv_ds_9')
        net = self._depthwise_separable_conv(net, 512, self.width_multiplier, sc='conv_ds_10')
        net = self._depthwise_separable_conv(net, 512, self.width_multiplier, sc='conv_ds_11')
        net = self._depthwise_separable_conv(net, 512, self.width_multiplier, sc='conv_ds_12')

        net = self._depthwise_separable_conv(net, 1024, self.width_multiplier, downsample=True, sc='conv_ds_13')
        net = self._depthwise_separable_conv(net, 1024, self.width_multiplier, sc='conv_ds_14')
        net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

        logits = slim.fully_connected(net, self.num_classes, activation=None, scope='fc_16')

        predictions = slim.softmax(logits, scope="predictions")

        return predictions


        