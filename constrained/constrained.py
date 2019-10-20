#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
linearly constrained optimization
1.Dual ascent method
2.Augmented Lagrangian
3.Projected gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt

INITIAL_LAMBDA = 0.0
INITIAL_X = np.array([0.0, 0.0]).T
C = 5.0


def g(x):
    return x[0] + x[1] - 1


def objective(x):
    return 3 * x[0] * x[0] + 2 * x[1] * x[1]


def dual_ascend_method():
    _lambda = INITIAL_LAMBDA
    epsilon = 1.0
    x_arr = []
    lambda_arr = [_lambda]
    for i in range(8):
        new_x = np.array([-_lambda / 6, -_lambda / 4]).T
        _lambda = _lambda + epsilon * g(new_x)
        x_arr.append(new_x)
        lambda_arr.append(_lambda)
    # print(x_arr)
    # print(lambda_arr)
    x_arr = np.array(x_arr)
    plt = plot_lagrange(x_arr, lambda_arr)
    plt.suptitle("Dual ascend method")
    plt.savefig("dual_ascend.png")


def augmented_lagrangian():
    def argmin(__lambda):
        _x = (C - __lambda) / (6 + 5 * C / 2)
        _y = (C - __lambda) / (4 + 5 * C / 3)
        return np.array([_x, _y]).T

    _lambda = INITIAL_LAMBDA
    x_arr = []
    lambda_arr = [_lambda]
    for i in range(8):
        new_x = argmin(_lambda)
        _lambda = _lambda + C * g(new_x)
        x_arr.append(new_x)
        lambda_arr.append(_lambda)
    # print(x_arr)
    # print(lambda_arr)
    x_arr = np.array(x_arr)
    plt = plot_lagrange(x_arr, lambda_arr)
    plt.suptitle("Augmented lagrangian method")
    plt.savefig("augmented_lagragian.png")


def projected_gradient_descent():
    def projection(x):
        _x = (1.0 + x[0] - x[1]) / 2.0
        _y = (1.0 - x[0] + x[1]) / 2.0
        return np.array([_x, _y]).T

    epsilon = 0.05
    new_x = np.array([0.0, 0.0]).T
    x_arr = [new_x]
    xhat_ar = [new_x]
    for i in range(8):
        xhat = new_x - epsilon * np.array([6 * new_x[0], 4 * new_x[1]]).T
        new_x = projection(xhat)
        x_arr.append(new_x)
        xhat_ar.append(xhat)
    # print(x_arr)
    # print(xhat_ar)
    x_arr = np.array(x_arr)
    xhat_ar = np.array(xhat_ar)
    plot_projection(x_arr, xhat_ar)


def plot_lagrange(points, lambdas):
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(121)
    X, Y = np.meshgrid(np.arange(0.0, 1.0, 0.1),
                       np.arange(0.0, 1.0, 0.1))
    Z = objective([X, Y])
    ax.contour(X, Y, Z, levels=20, cmap='spring_r')
    ax.plot(points[:, 0], points[:, 1])
    ax.set_xlim([0, 0.7])
    ax.set_ylim([0, 0.7])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    x = np.arange(0, 3)
    y = 1 - x
    ax.plot(x, y, color="red")
    for i in range(len(points)):
        ax.text(points[i, 0], points[i, 1], '%s' % (str(i)), size=10, zorder=1,
                color='k')
    ax = fig.add_subplot(122)
    ax.plot(lambdas)
    ax.set_xlabel("iteration")
    ax.set_ylabel("$\lambda$")
    ax.set_ylim([-3.0, 1.0])

    return plt


def plot_projection(points, outers):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(np.arange(0.0, 1.0, 0.1),
                       np.arange(0.0, 1.0, 0.1))
    Z = objective([X, Y])
    ax.contour(X, Y, Z, levels=20, cmap='spring_r')
    ax.plot(points[:, 0], points[:, 1], color='b')
    ax.plot(outers[:, 0], outers[:, 1], color='c')
    ax.set_xlim([0, 0.7])
    ax.set_ylim([0, 0.7])
    x = np.arange(0, 3)
    y = 1 - x
    ax.plot(x, y, color="red")
    for i in range(len(points)):
        ax.text(points[i, 0], points[i, 1], '%s' % (str(i)), size=10, zorder=1,
                color='k')
        ax.text(outers[i, 0], outers[i, 1], '%s' % (str(i)), size=10, zorder=1,
                color='k')
    plt.suptitle("Projected Gradient Descent method")
    plt.savefig("projecetd_gradient_descent.png")
    return


if __name__ == '__main__':
    dual_ascend_method()
    augmented_lagrangian()
    projected_gradient_descent()
