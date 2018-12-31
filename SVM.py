import argparse
from utils.GD_model import GD_model
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

    # read file
    train_x, train_y = _read_data(train_data)
    test_x, test_y = _read_data(test_data)

    process_end = time.time()

    time_limit = process_end - start_time - 1

    # training
    model = GD_model(train_x, train_y, time_limit)
    model.train()
    predict_y = model.predict(test_x)

    # evaluate
    acc = compare_answer(predict_y, test_y)
    print(acc)


def _read_data(file):
    content = [line.strip() for line in file.readlines()]
    x = []
    y = []
    for line in content:
        temp = line.split()
        xi = temp[:-1]
        yi = temp[-1]
        x.append([float(x) for x in xi])
        y.append(float(yi))
    return np.array(x), np.array(y)


def compare_answer(y_1, y_2):
    assert len(y_1) == len(y_2)
    length = len(y_1)
    corr = 0
    for i in range(length):
        if y_1[i] == y_2[i]:
            corr += 1
    return corr / length


if __name__ == '__main__':
    main()
