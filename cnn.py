# _*_ coding:utf-8 _*_
"""
file: cnn.py
date: 2018-12-19 20:33
author: G
desc:

"""

import tensorflow as tf


def inference(images, batch_size, n_classes):
    """

    :param images:      图像
    :param batch_size:  批数量
    :param n_classes:   类别数量
    :return:
    """
    # 在conv1作用域下创建相关变量
    with tf.variable_scope('conv1') as scope:
        # 创建变量weights
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        # 创建变量biases
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')

        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    # 在作用域pooling_lrn下创建变量
    with tf.variable_scope('pooling_lrn') as scope:
        # 使用最大化池化层
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')
    # 在作用域conv2下创建变量
    with tf.variable_scope('conv2') as scope:
        # 创建weights变量
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 16, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        # 创建biases变量
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 初始化当前卷积层
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        # 添加偏置
        pre_activation = tf.nn.bias_add(conv, biases)
        # 添加激活函数
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    # 在pooling2_lrn创建参数
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        # 使用最大化池化层
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')
    # 在local3作用域下创建变量
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    # 在local4作用域下创建变量
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(stddev=0.005, dtype=tf.flfloat32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
    # 在softmax_linear作用域下创建变量
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    return softmax_linear
