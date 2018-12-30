import numpy as np
import time


class GD:
    def __init__(self, start_time, limit_time, x, y, epochs=10000, learning_rate=0.01):
        self.start_time = start_time
        self.limit_time = limit_time
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size=np.shape(self.x)[1])

    def get_loss(self, x, y):
        loss = max(0, 1 - y * np.dot(x, self.w))
        return loss

    def cal_sgd(self, x, y, w):
        if y * np.dot(x, w) < 1:
            w = w - self.learning_rate * (-y * x)
        else:
            w = w
        return w

    def train(self):
        count = 0
        while True:
            count += 1
            if time.time() - self.start_time > self.limit_time-1\
                    or count >= self.epochs:
                break
            randomize = np.arange(len(self.x))
            np.random.shuffle(randomize)  # 打乱顺序
            x = self.x[randomize]
            y = self.y[randomize]
            loss = 0
            for xi, yi in zip(x, y):  # 打包tuple
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w)
            # print("epoch:{0} loss:{1}".format(count, loss))

    def predict(self, x):
        x_test = np.c_[np.ones((x.shape[0])), x]
        return np.sign(np.dot(x_test, self.w))


