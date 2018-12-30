import argparse
from GD import GD
import numpy as np
import time


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="SVM Problem")
    parser.add_argument('train', metavar='train data', type=argparse.FileType('r'))
    parser.add_argument('test', metavar='test data', type=argparse.FileType('r'))
    parser.add_argument('-t', metavar='time budget', type=int)
    arg = parser.parse_args()
    train_file = arg.train
    test_file = arg.test
    # read file
    x, y = _test_read(train_file, 0, 600)
    test_data, test_answer = _test_read(test_file, 601, 1334)
    #  ============== GD ==============
    gd = GD(start_time, arg.t, x, y)
    gd.train()
    predicted = gd.predict(test_data)
    # =============compare ==========
    accu = compare_answer(predicted, test_answer)
    print(accu)


def read_data(file):
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


def _test_read(file, begin, end):
    content = [line.strip() for line in file.readlines()]
    x = []
    y = []
    count = 0
    for line in content:
        if begin <= count <= end:
            temp = line.split()
            xi = temp[:-1]
            yi = temp[-1]
            x.append([float(x) for x in xi])
            y.append(float(yi))
        count += 1
    return np.array(x), np.array(y)


def compare_answer(answer1, answer2):
    if len(answer1) == len(answer2):
        corr = 0
        for i in range(len(answer1)):
            if answer1[i] == answer2[i]:
                corr += 1
        return corr / len(answer1)
    else:
        print('answer length not correct')


if __name__ == '__main__':
    main()
