#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Gauss kernel mode k-fold cross-validation
for band width h and regularization parameter lambda
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [16, 9]
np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


def fit(x, y, l, h):
    """

    :param x:
    :param y:
    :param l:regularization
    :param h:band
    :return:
    """
    # calculate design matrix
    k = calc_design_matrix(x, x, h)
    # print("k.shape", k.shape)
    # print("y.shape", y.shape)
    # solve the least square problem
    A = k.T.dot(k) + l * np.identity(len(k))
    b = k.T.dot(y[:, None])
    # print("A.shape", A.shape)
    # print("b.shape", b.shape)
    theta = np.linalg.solve(A, b)
    return k, theta


# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)
# k-fold cross-validation
best_l, best_h = None, None
Score = 100000000
n_fold = 50
X = np.linspace(start=xmin, stop=xmax, num=5000)
h_list = [0.03, 0.3, 3.0]
l_list = [0.0001, 0.1, 100.0]
# implement here
for h_iter in range(3):
    for l_iter in range(3):
        score = 0
        h = h_list[h_iter]
        l = l_list[l_iter]
        for fold in range(n_fold):
            index = np.ones(sample_size, dtype=bool)
            index[fold * 1:(fold + 1) * 1] = False
            x_train = x[index]
            y_train = y[index]
            x_valid = x[np.invert(index)]
            y_valid = y[np.invert(index)]
            # print(y_train.shape)
            # assert x_valid.shape[0] == 10, x_valid.shape
            k, theta = fit(x_train, y_train, l, h)
            K = calc_design_matrix(x_train, x_valid, h)
            prediction = K.dot(theta)
            score += np.linalg.norm(y_valid - prediction)
        print('(l, h): ({}, {}), Score: {}'.format(l, h, score))
        if Score > score:
            best_l = l
            best_h = h
            Score = score
        #
        # visualization
        #
        # calculate design matrix
        k = calc_design_matrix(x, x, h)

        # solve the least square problem
        theta = np.linalg.solve(
            k.T.dot(k) + l * np.identity(len(k)),
            k.T.dot(y[:, None]))

        # create data to visualize the prediction
        K = calc_design_matrix(x, X, best_h)
        prediction = K.dot(theta)
        plt.xlim(-3, 3)
        plt.ylim(-1, 1.5)
        plt.subplot(3, 3, l_iter + 1 + 3 * h_iter)
        plt.scatter(x, y, c='green', marker='o')
        plt.title('($\lambda, h$): ({}, {})'.format(l, h))
        plt.plot(X, prediction)

plt.savefig('ML1-homework1.png')
print('Best (l, h): ({}, {})'.format(best_l, best_h))
