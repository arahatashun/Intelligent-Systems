#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata

import numpy as np
import matplotlib.pyplot as plt


def chi(deg_free):
    """ｓ

    :param deg_free:  degree of freedom
    :return:
    """
    x = np.random.normal(size=[1000, deg_free])
    y = np.sum(x * x, axis=1)
    return y


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(1, 6):
        res = chi(i)
        ax.hist(res, density=True, alpha=0.3, label=str(i))
    ax.set_xlabel('x')
    ax.set_ylabel('freq')
    ax.legend()
    plt.title("Chi squared")
    plt.show()
