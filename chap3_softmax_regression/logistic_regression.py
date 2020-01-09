import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(z,y):
    (n,m) = y.shape
    return 1/n*np.sum(z-y)**2

def gradient_descent(x, y, learn_rate=0.1, iterations=5000):
    (n, m) = x.shape
    w = np.zeros((m,1))
    
    loss_history = np.zeros(iterations)
    for iter in range(iterations):
        z = sigmoid(np.dot(x, w))
        w = w - learn_rate / n * np.dot(x.T, z - y)
        loss_history[iter] = loss(z,y)
    
    return w

def compute(x,w):
    w1 = w[0][0]
    w2 = w[1][0]
    b = w[2][0]
    return -(b+w1*x)/w2
    
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
    
    x = data_set[:,0:2]
    x = np.insert(x,len(x[0]),1,1)
    
    y = data_set[:,2:]
    
    w = gradient_descent(x,y)
    
#    plt.show()
    
    x_res = np.arange(0,8,0.1)
    y_res = np.array([compute(i,w) for i in x_res])
    
    plt.plot(x_res,y_res)
    
    plt.show()
