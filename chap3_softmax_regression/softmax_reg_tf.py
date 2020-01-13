import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


dot_num = 100
x_p = np.random.normal(3., 1, dot_num)
y_p = np.random.normal(6., 1, dot_num)
y = np.ones(dot_num)
C1 = np.array([x_p, y_p, y]).T

x_n = np.random.normal(6., 1, dot_num)
y_n = np.random.normal(3., 1, dot_num)
y = np.zeros(dot_num)
C2 = np.array([x_n, y_n, y]).T

x_b = np.random.normal(7., 1, dot_num)
y_b = np.random.normal(7., 1, dot_num)
y = np.ones(dot_num)*2
C3 = np.array([x_b, y_b, y]).T

plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')

data_set = np.concatenate((C1, C2, C3), axis=0)
np.random.shuffle(data_set)

epsilon = 1e-12

epsilon = 1e-12


class SoftmaxRegression():
    def __init__(self):
        '''============================='''
        # todo 填空一，构建模型所需的参数 self.W, self.b 可以参考logistic-regression-exercise
        '''============================='''
        self.W = tf.Variable(shape=[2, 3], dtype=tf.float32,
                             initial_value=tf.random.uniform(shape=[2, 3], minval=-0.1, maxval=0.1))

        self.b = tf.Variable(shape=[1, 3], dtype=tf.float32, initial_value=tf.zeros(shape=[1, 3]))

        self.trainable_variables = [self.W, self.b]
    
    @tf.function
    def __call__(self, inp):
        logits = tf.matmul(inp, self.W) + self.b  # shape(N, 3)
        pred = tf.nn.softmax(logits)
        return pred


@tf.function
def compute_loss(pred, label):
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), dtype=tf.float32, depth=3)
    '''============================='''
    # 输入label shape(N, 3), pred shape(N, 3)
    # 输出 losses shape(N,) 每一个样本一个loss
    # todo 填空二，实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)
    '''============================='''
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = label,logits = pred)
    
    loss = tf.reduce_mean(losses)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, axis=1), tf.argmax(pred, axis=1)), dtype=tf.float32))
    return loss, accuracy


@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = compute_loss(pred, y)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy

if __name__ == '__main__':
    model = SoftmaxRegression()
    opt = tf.keras.optimizers.SGD(lr=0.1)
    x1, x2, y = list(zip(*data_set))
    x = list(zip(x1, x2))
    for i in range(10000):
        loss, accuracy = train_one_step(model, opt, x, y)
        if i % 500 == 499:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')
            
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')

    x = np.arange(0., 10., 0.1)
    y = np.arange(0., 10., 0.1)

    X, Y = np.meshgrid(x, y)
    inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
    print(inp.shape)
    Z = model(inp)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z)
    plt.show()