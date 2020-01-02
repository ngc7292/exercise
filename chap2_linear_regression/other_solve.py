import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret


def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    # ==========
    # todo '''请实现多项式基函数'''
    # ==========
    ret = None
    for i in range(3):
       ret = np.concatenate((ret, x ** i), 1) if i != 0 else x ** i

    return ret


def gaussian_basis(x, feature_num=10):
    '''高斯基函数'''
    # ==========
    # todo '''请实现高斯基函数'''
    # ==========
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)

    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    ret = np.insert(ret,0,1,1)
    return ret


def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

def loss(x, y, theta):
    m = np.size(x[:, 0])
    return 1 / (2 * m) * np.sum((np.dot(x, theta) - y) ** 2)

def main(x_train, y_train, basic_function=None, iterations=200000, alpha=0.1):
    basic_function = gaussian_basis if basic_function is None else basic_function
    x_train = basic_function(x_train)
    y_train = np.reshape(y_train, (len(y_train),1))
    theta = np.zeros((len(x_train[0]), 1))
    m = len(x_train)
    loss_history = np.zeros((iterations, 1))
    for i in range(iterations):
        theta = theta - alpha / m * np.dot(x_train.T,(np.dot(x_train,theta) - y_train))
        loss_history[i] = loss(x_train,y_train,theta)

    plt.figure()
    plt.plot(range(iterations), loss_history)
    plt.show()

    def f(X_test):
        X_test = basic_function(X_test)
        return np.dot(X_test, theta)

    return f


# 程序主入口（建议不要改动以下函数的接口）
if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train[:10])
    print(x_test.shape)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
    f = main(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

    # 计算预测的输出值
    y_test_pred = f(x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))

    # 显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3)
    #     plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()
