import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SVM Problem")
    parser.add_argument('train', metavar='train data', type=argparse.FileType('r'))
    parser.add_argument('test', metavar='test data', type=argparse.FileType('r'))
    parser.add_argument('-t', metavar='time budget', type=int)
    arg = parser.parse_args()
    train_file = arg.train
    test_file = arg.test
    # read file
    train_x, train_y = _read_data(train_file)
    test_x, test_y = _read_data(test_file)
    # training and predict
    gd = model(train_x, train_y)
    gd.train()
    pred_y = gd.predict(test_x)
    # evaluate
    acc = evaluator(pred_y, test_y)
    print(acc)


def _read_data(file):
    x = list()
    y = list()
    for line in file.readlines():
        temp = line.strip().split()
        xi = temp[:-1]
        yi = temp[-1]
        x.append([float(x) for x in xi])
        y.append(float(yi))
    return np.array(x), np.array(y)


def evaluator(predict, test):
    assert len(predict) == len(test)
    corr = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            corr += 1
    return corr / len(predict)


class model:
    def __init__(self, x, y, epochs=200, learning_rate=0.01):
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size=np.shape(self.x)[1], )

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
        for epoch in range(self.epochs):
            randomize = np.arange(len(self.x))
            np.random.shuffle(randomize)
            x = self.x[randomize]
            y = self.y[randomize]
            loss = 0
            for xi, yi in zip(x, y):
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w)
                # print('epoch: {0} loss: {1}'.format(epoch, loss))

    def predict(self, x):
        x_test = np.c_[np.ones((x.shape[0])), x]
        return np.sign(np.dot(x_test, self.w))


if __name__ == '__main__':
    main()
