#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, models
import os, sys


def gen_data_batch(batch_size, start, end):
    '''在(start, end)区间采样生成一个batch的整型的数据
    Args :
        batch_size: batch_size
        start: 开始数值
        end: 结束数值
    '''
    numbers_1 = np.random.randint(start, end, batch_size)
    numbers_2 = np.random.randint(start, end, batch_size)
    results = numbers_1 + numbers_2
    return numbers_1, numbers_2, results


def convertNum2Digits(Num):
    '''将一个整数转换成一个数字位的列表,例如 133412 ==> [1, 3, 3, 4, 1, 2]
    '''
    strNum = str(Num)
    chNums = list(strNum)
    digitNums = [int(o) for o in strNum]
    return digitNums


def convertDigits2Num(Digits):
    '''将数字位列表反向， 例如 [1, 3, 3, 4, 1, 2] ==> [2, 1, 4, 3, 3, 1]
    '''
    digitStrs = [str(o) for o in Digits]
    numStr = ''.join(digitStrs)
    Num = int(numStr)
    return Num


def pad2len(lst, length, pad=0):
    '''将一个列表用`pad`填充到`length`的长度 例如 pad2len([1, 3, 2, 3], 6, pad=0) ==> [1, 3, 2, 3, 0, 0]
    '''
    lst += [pad] * (length - len(lst))
    return lst


def results_converter(res_lst):
    '''将预测好的数字位列表批量转换成为原始整数
    Args:
        res_lst: shape(b_sz, len(digits))
    '''
    res = [reversed(digits) for digits in res_lst]
    return [convertDigits2Num(digits) for digits in res]


def prepare_batch(Nums1, Nums2, results, maxlen):
    '''准备一个batch的数据，将数值转换成反转的数位列表并且填充到固定长度
    Args:
        Nums1: shape(batch_size,)
        Nums2: shape(batch_size,)
        results: shape(batch_size,)
        maxlen:  type(int)
    Returns:
        Nums1: shape(batch_size, maxlen)
        Nums2: shape(batch_size, maxlen)
        results: shape(batch_size, maxlen)
    '''
    Nums1 = [convertNum2Digits(o) for o in Nums1]
    Nums2 = [convertNum2Digits(o) for o in Nums2]
    results = [convertNum2Digits(o) for o in results]
    
    Nums1 = [list(reversed(o)) for o in Nums1]
    Nums2 = [list(reversed(o)) for o in Nums2]
    results = [list(reversed(o)) for o in results]
    
    Nums1 = [pad2len(o, maxlen) for o in Nums1]
    Nums2 = [pad2len(o, maxlen) for o in Nums2]
    results = [pad2len(o, maxlen) for o in results]
    
    return Nums1, Nums2, results


class myRNNModel(keras.Model):
    def __init__(self):
        super(myRNNModel, self).__init__()
        self.embed_layer = tf.keras.layers.Embedding(10, 32, batch_input_shape=[None, None])
        self.rnncell = tf.keras.layers.SimpleRNNCell(64)
        self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True)
        self.dense = tf.keras.layers.Dense(10)
    
    @tf.function
    def __call__(self, num1, num2):
        '''
        此处完成上述图中模型
        '''
        x_1 = self.embed_layer(num1)
        x_2 = self.embed_layer(num2)
        output = layers.concatenate([x_1,x_2])
        rnn_r = self.rnn_layer(output)
        logits = self.dense(rnn_r)
        
        return logits

@tf.function
def compute_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    return tf.reduce_mean(losses)


@tf.function
def train_one_step(model, optimizer, x, y, label):
    with tf.GradientTape() as tape:
        logits = model(x, y)
        loss = compute_loss(logits, label)
    
    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(steps, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=555555555)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
        x = tf.constant(Nums1, dtype=tf.int32)
        y = tf.constant(Nums2, dtype=tf.int32)
        
        label = tf.constant(results, dtype=tf.int32)
        
        loss = train_one_step(model, optimizer, x, y, label)
        
        if step % 50 == 0:
            print('step', step, ': loss', loss.numpy())
    
    return loss

def train_2(steps):
    datas = gen_data_batch(200,0,555555555)
    x_1, x_2, results = prepare_batch(*datas, maxlen=11)
    
    print(1)

def evaluate(model):
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    logits = model(tf.constant(Nums1, dtype=tf.int32), tf.constant(Nums2, dtype=tf.int32))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:20]:
        print(o[0], o[1], o[0] == o[1])
    
    print('accuracy is: %g' % np.mean([o[0] == o[1] for o in zip(datas[2], res)]))


if __name__ == '__main__':
    optimizer = optimizers.Adam(0.001)
    model = myRNNModel()
    
    train(3000, model, optimizer)
    evaluate(model)

    # train_2(5000)
    