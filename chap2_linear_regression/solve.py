import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt("train.txt",delimiter="\t")
x = data[:,0]
y = data[:,1]
# print(data[:10])
# print(x[:10])

# plt.figure()
#plt.scatter(x,y)
# plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(x, y, theta):
    m = np.size(x[:, 0])
    return 1 / (2 * m) * np.sum((np.dot(x, theta) - y) ** 2)


def linner_reg_fit(x, y,  iterations=1500, alpha=0.01):
    m = len(x)
    x = np.insert(x, len(x[0]), 1, 1)
    theta = np.zeros((len(x[0]), 1))


    loss_history = np.zeros((iterations, 1))
    for item in range(iterations):
        theta = theta - alpha / m * np.dot(x.T, (np.dot(x, theta) - y))
        loss_history[item] = loss(x, y, theta)

    return theta,loss_history

x = np.reshape(x,(len(x),1))
y = np.reshape(y,(len(y),1))


theta,loss_history = linner_reg_fit(x,y)


x_insert = np.insert(x, len(x[0]), 1, 1)

plt.figure(1)
plt.plot(x, np.dot(x_insert, theta), '-')
plt.scatter(x, y)

plt.figure(2)
plt.plot(range(1500),loss_history)

plt.show()