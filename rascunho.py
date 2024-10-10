import numpy as np
import os

def normalize1_row(data):

    data_norm = np.copy(data)

    # print(data)
    # print(data_norm)

    for i, row in enumerate(data):
        # print(i)
        # print(row)
        # print(data[i])
        # print(np.sum(data[i]))
        data_norm[i] = data[i]/np.sum(data[i])
        # print(data_norm[i])

    return data_norm

if __name__ == "__main__":

    b = np.array([[0., 10., 3., 4.],[1., 5., 6., 2.]])
    p = np.array([[0., 0., 1., 0.],[1., 0., 0., 0.]])

    b2 = normalize1_row(b)

    b_cumsum = np.cumsum(b2,1)

    # print(b)
    # print(b2)
    print(b_cumsum)

    r = 0.3

    print(b_cumsum >= r)
    # print(np.where(b_cumsum >= r))

    # for i, row in enumerate(b_cumsum):
    for i, row in enumerate(p):
        print(np.where(row >= r)[0][0])