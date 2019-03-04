import argparse
from utils.GD_model import GD_model
from utils.SMO_model import SMO_model
import numpy as np
import time


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="SVM Problem")
    parser.add_argument('train', metavar='train data', type=argparse.FileType('r'))
    parser.add_argument('test', metavar='test data', type=argparse.FileType('r'))
    parser.add_argument('-t', metavar='time budget', type=int)
    arg = parser.parse_args()
    train_data = arg.train
    test_data = arg.test
    time_limit = arg.t

    # read file
    train_x, train_y = _read_data(train_data)
    test_x, test_y = _read_data(test_data)

    process_end = time.time()

    time_limit = time_limit - (process_end - start_time)
    # print(time_limit)
    # for iteration_times in [20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
    # training
    iteration_times = 200
    gd_start = time.time()
    model = GD_model(train_x, train_y, time_limit)
    model.train(iteration_times)
    predict_y_GD = model.predict(test_x)
    gd_end = time.time()
    # ----- SMO ------
    # smo_start = time.time()
    # model = SMO_model()
    # model.fit(train_x, train_y, iteration_times)
    # predict_y_SMO = model.predict(test_x)
    # smo_end = time.time()
    # evaluate
    acc_GD = _evaluate(predict_y_GD, test_y)
    # acc_SMO = _evaluate(predict_y_SMO, test_y)
    # print('iteration_times: {}'.format(iteration_times))
    # print('GD: {}, time cost: {}'.format(acc_GD, gd_end-gd_start))
    for y in predict_y_GD:
        print(y)


def _read_data(file):
    x, y = list(), list()
    for line in file.readlines():
        temp = line.strip().split()
        xi = temp[:-1]
        yi = temp[-1]
        x.append([float(x) for x in xi])
        y.append(float(yi))
    return np.array(x), np.array(y)


def _evaluate(y_1, y_2):
    assert len(y_1) == len(y_2)
    length = len(y_1)
    corr = 0
    for i in range(length):
        if y_1[i] == y_2[i]:
            corr += 1
    return corr / length


if __name__ == '__main__':
    main()
