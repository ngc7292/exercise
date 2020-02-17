import tensorflow as tf

import numpy as np


def softmax(x):
    ##########
    '''实现softmax函数，只要求对最后一维归一化，
    不允许用tf自带的softmax函数'''
    ##########
    x = np.exp(x)
    prob_x = []
    for items in x:
        sum = np.sum(items)
        prob_x.append(items / sum)
    return tf.add(prob_x, 0)


def sigmoid(x):
    ##########
    '''实现sigmoid函数， 不允许用tf自带的sigmoid函数'''
    ##########
    return tf.add(1 / (1 + np.exp(-x)), 0)


def sigmoid_ce(pred, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    res = tf.add(
        tf.subtract(
            tf.math.maximum(pred, [0]),
            tf.multiply(pred, label)
        ),
        tf.math.log(1 + tf.math.exp(-tf.abs(pred)))
    )
    return res

def softmax_ce(pred, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    softmax = tf.exp(pred - tf.reduce_max(pred)) / tf.reduce_sum(tf.exp(pred - tf.reduce_max(pred)))
    return -tf.reduce_sum(tf.multiply(label, softmax))


if __name__ == '__main__':
    # test_data = np.random.normal(size=[10, 5])
    # print((softmax(test_data).numpy() - tf.nn.softmax(test_data, axis=-1).numpy()) ** 2 < 0.0001)
    # print(softmax(test_data))
    # print(tf.nn.softmax(test_data,axis = -1))
    
    # print(sigmoid(test_data))
    # print(tf.nn.sigmoid(test_data))
    # print((sigmoid(test_data).numpy() - tf.nn.sigmoid(test_data).numpy()) ** 2 < 0.0001)
    
    # test_data = np.random.normal(size=[10])
    # prob = tf.nn.sigmoid(test_data)
    # label = np.random.randint(0, 2, 10).astype(test_data.dtype)
    # print(tf.nn.sigmoid_cross_entropy_with_logits(label, test_data))
    # print(sigmoid_ce(test_data, label))
    
    # print((tf.nn.sigmoid_cross_entropy_with_logits(label, test_data) - sigmoid_ce(test_data, label) < 0.001))
    test_data = np.random.normal(size=[10])
    prob = tf.nn.sigmoid(test_data)
    label = np.random.randint(0, 2, 10).astype(test_data.dtype)
    print(label)