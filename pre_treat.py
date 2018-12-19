# _*_ coding:utf-8 _*_
"""
file: pre_treat.py
date: 2018-12-19 11:03
author: G
desc:

"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 导入必要的包



def get_files(file_dir):
    """
    读取训练集文件函数
    :param file_dir: 文件路径
    :return: image_list 图像数组
              label_list 标签数组
    """
    # 训练集数组
    A5 = []
    # 训练集标签计数数组
    label_A5 = []
    A6 = []
    label_A6 = []
    SEG = []
    label_SEG = []
    SUM = []
    label_SUM = []
    LTAX1 = []
    label_LTAX1 = []
    # 从文件路径逐个读取文件并按照名称存入相应的数组
    # 这里一定要注意，如果是多分类问题的话，一定要将分类的标签从0开始。这里是五类
    # 标签为0，1，2，3，4。我之前以为这个标签应该是随便设置的，结果就出现了Target[0] out of range的错误。
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'A5':
            A5.append(file_dir + file)
            label_A5.append(1)
        elif name[0] == 'A6':
            A6.append(file_dir + file)
            label_A6.append(2)
        elif name[0] == 'LTAX1':
            LTAX1.append(file_dir + file)
            label_LTAX1.append(3)
        elif name[0] == 'SEG':
            SEG.append(file_dir + file)
            label_SEG.append(4)
        else:
            SUM.append(file_dir + file)
            label_SUM.append(5)
            # 验证检查文件读取情况
    print('There are %d A5\nThere are %d A6\nThere are %d LTAX1\nThere are %d SEG\nThere are %d SUM' \
          % (len(A5), len(A6), len(LTAX1), len(SEG), len(SUM)))
    # 将所有文件统一起来,用来水平合并数组
    image_list = np.hstack((A5, A6, LTAX1, SEG, SUM))
    label_list = np.hstack((label_A5, label_A6, label_LTAX1, label_SEG, label_SUM))
    # 将文件和标签整合
    temp = np.array([image_list, label_list])
    # 将tmep进行转置
    temp = temp.transpose()
    # 将temp进行随机变换
    np.random.shuffle(temp)
    # 将随机变换后的文件和标签存储
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    # 返回文件和标签数组
    return image_list, label_list



def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    获得图像和标签队列
    返回两个batch，两个batch即为传入神经网络的数据
    :param image: 输入图像
    :param label: 输入标签
    :param image_W: 图像宽
    :param image_H: 图像高
    :param batch_size: 批数量
    :param capacity:   容量
    :return:   image_batch 图像batch
                label_batch 标签batch
    """
    # 将图像转成string格式
    image = tf.cast(image, tf.string)
    # 将标签转换成int格式
    label = tf.cast(label, tf.int32)
    # 生成文件名队列
    input_queue = tf.train.slice_input_producer([image, label])

    # 返回进队的标签
    label = input_queue[1]
    # 返回进队的图片文件
    image_contents = tf.read_file(input_queue[0])
    # 对jpg图像进行解码
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 将图像进行填充或者裁剪到同意大小
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 将图像特征进行标准化
    image = tf.image.per_image_standardization(image)
    # 将图像进行tensor队列生成
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=8, capacity=capacity)
    # 将标签重构
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


# 分类类别个数
BATCH_SIZE = 5
CAPACITY = 64
IMG_W = 208
IMG_H = 208

train_dir = 'C:/Users/Administrator/Documents/PycharmProj/Demo_5class/train/'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# 执行入队线程管理器
with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 2:
            # 提取出两个batch的图片并可视化。
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print('label: %d' % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
