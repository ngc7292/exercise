# python: 3.5.2
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """
    
    def __init__(self):
        # 请补全此处代码
        self.learning_rate = 0.01
    
    def train(self, data, epoch=10000):
        """
        训练模型。
        """
        global val1
        x = data[:, 0:2]
        x = np.insert(x, len(x[0]), 1, 1)
        y = data[:, 2]
        self.w = np.zeros(len(x[0]))
        
        for e in range(epoch):
            out = []
            for i, val in enumerate(x):
                val1 = np.dot(x[i], self.w)
                out.append(1) if val1 >= 0 else out.append(-1)
                if val1 * y[i] < 1:
                    self.w = self.w + self.learning_rate * ((y[i] * x[i]) - (2 * (1 / epoch) * self.w))
                else:
                    self.w = self.w + self.learning_rate * (-2 * 1 / epoch * self.w)
            out = np.array(out)
            acc = eval_acc(y, out)
            
            if e % 500 == 0:
                print(f"{e} times acc: {acc}\n")
        
        # 请补全此处代码
    
    def predict(self, x):
        """
        预测标签。
        """
        x = np.insert(x, len(x[0]), 1, 1)
        
        res = []
        for x_i in x:
            res.append(1) if np.dot(x_i, self.w) >= 0 else res.append(-1)
        return res
        
        # 请补全此处代码
    
    def compute(self, x1):
        w1 = self.w[0]
        w2 = self.w[1]
        b = self.w[2]
        return (-b - w1 * x1) / w2


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)
    # print(data_train[:10])
    
    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型
    
    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    for val, inp in enumerate(x_train):
        if t_train[val] == 1:
            plt.scatter(inp[0], inp[1], s=100, marker='_', linewidths=5)
        else:
            plt.scatter(inp[0], inp[1], s=100, marker='+', linewidths=5)
    
    x = [i for i in range(-40,120,1)]
    y = [svm.compute(i) for i in range(-40, 120, 1)]
    plt.plot(x,y)
    
    plt.show()
    
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)
    
    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
