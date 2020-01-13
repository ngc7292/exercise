import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression():
    def __init__(self):
        self.W = tf.Variable(shape=[2, 1], dtype=tf.float32,
                             initial_value=tf.random.uniform(shape=[2, 1], minval=-0.1, maxval=0.1))
        
        self.b = tf.Variable(shape=[1], dtype=tf.float32, initial_value=tf.zeros(shape=[1]))
        
        self.trainable_variables = [self.W, self.b]
    
    @tf.function
    def __call__(self, inp):
        logits = tf.matmul(inp, self.W) + self.b  # shape(N, 1)
        pred = tf.nn.sigmoid(logits)
        return pred


@tf.function
def compute_loss(pred, label):
    if not isinstance(label, tf.Tensor):
        label = tf.constant(label, dtype=tf.float32)
    pred = tf.squeeze(pred, axis=1)
    '''============================='''
    # 输入label shape(N,), pred shape(N,)
    # 输出 losses shape(N,) 每一个样本一个loss
    # todo 填空一，实现sigmoid的交叉熵损失函数(不使用tf内置的loss 函数)
    '''============================='''

    # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
    losses = cross_cost(pred, label)
    loss = tf.reduce_mean(losses)
    
    pred = tf.where(pred > 0.5, tf.ones_like(pred), tf.zeros_like(pred))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pred), dtype=tf.float32))
    return loss, accuracy


@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = compute_loss(pred, y)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy, model.W, model.b


def compute(w, b, x):
    w1 = w[0]
    w2 = w[1]
    return (-w1 * x - b) / w2


def sigmoid(z):
    return 1 / (1 + tf.exp(-z))


def cross_cost(pred, label):
    # 参考公式 coss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # res = tf.add(tf.multiply(-label, (-tf.math.log(sigmoid(pred)))), tf.multiply((1 - label), (-tf.math.log(1 - sigmoid(pred)))))
    res = tf.add(
        tf.subtract(
            tf.math.maximum(pred,[0]),
            tf.multiply(pred,label)
        ),
        tf.math.log(1+tf.math.exp(-tf.abs(pred)))
    )
    return res


if __name__ == '__main__':
    dot_num = 100
    x_p = np.random.normal(3., 1, dot_num)
    y_p = np.random.normal(6., 1, dot_num)
    y = np.ones(dot_num)
    C1 = np.array([x_p, y_p, y]).T
    
    x_n = np.random.normal(6., 1, dot_num)
    y_n = np.random.normal(3., 1, dot_num)
    y = np.zeros(dot_num)
    C2 = np.array([x_n, y_n, y]).T
    
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    
    data_set = np.concatenate((C1, C2), axis=0)
    np.random.shuffle(data_set)
    
    epsilon = 1e-12
    
    model = LogisticRegression()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    x1, x2, y = list(zip(*data_set))
    
    x = list(zip(x1, x2))
    animation_fram = []
    
    for i in range(2000):
        
        loss, accuracy, W_opt, b_opt = train_one_step(model, opt, x, y)
        animation_fram.append((W_opt.numpy()[0, 0], W_opt.numpy()[1, 0], b_opt.numpy(), loss.numpy()))
        if i % 1000 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')
    
    x_data = np.arange(0, 8, 0.1)
    w = W_opt.numpy()
    b = b_opt.numpy()
    y_data = [compute(w, b, i) for i in x_data]
    
    plt.plot(x_data, y_data)
    plt.show()
