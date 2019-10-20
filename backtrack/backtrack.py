#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Backtracking line search and fixed step size comparison
"""
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

ALPHA = 0.5
BETA = 0.8
EPSILON = 1.0


def object(x):
    res = 10.0 * x[0] * x[0] + x[1] * x[1]
    # print("x:", x, " obj:", res)
    return res


def nabla(x):
    res = np.array([20.0 * x[0], 2 * x[1]]).T
    # print("x:", x, " grad:", res)
    return res


def armijo(x, epsilon):
    """ armijo condition
    :param x:
    :param epsilon:
    :return:
    """
    grad = nabla(x)
    left = object(x - epsilon * grad) - object(x)
    right = - ALPHA * epsilon * grad.T @ grad
    res = left.item() <= right.item()
    # print("left:", left, "right:", right, "res:", res)
    assert type(res) is bool, type(res)
    return res


def backtrack(x):
    eps = EPSILON
    while armijo(x, eps) is False:
        eps = BETA * eps
    return eps


def plot(points, objs):
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(np.arange(-5., 5., 0.5),
                       np.arange(-5., 5., 0.5))
    Z = object([X, Y])
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

    ax.set_xlabel('X')
    ax.set_xlim(-5, 5)
    ax.set_ylabel('Y')
    ax.set_ylim(-5, 5)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 300)
    ax.plot(points[:, 0], points[:, 1], objs)
    for i in range(len(objs)):
        ax.text(points[i, 0], points[i, 1], objs[i], '%s' % (str(i)), size=5, zorder=1,
                color='k')
    ax = fig.add_subplot(122)
    ax.contour(X, Y, Z, levels=20, cmap='spring_r')
    ax.plot(points[:, 0], points[:, 1])
    for i in range(len(objs)):
        ax.text(points[i, 0], points[i, 1], '%s' % (str(i)), size=5, zorder=1,
                color='k')
    return plt


if __name__ == '__main__':
    xinit = np.array([4.0, 2.0]).T
    x = xinit
    x_arr = []
    obj_arr = []
    for i in range(5):
        x_arr.append(x)
        obj_arr.append(object(x))
        x = x - backtrack(x) * nabla(x)
    res = plot(np.array(x_arr), np.array(obj_arr))
    res.title("backtracking line search")
    res.savefig("backtrack.png")
    # fixed step size
    x = xinit
    x_arr = []
    obj_arr = []
    for i in range(5):
        x_arr.append(x)
        obj_arr.append(object(x))
        x = x - 0.05 * nabla(x)
    res = plot(np.array(x_arr), np.array(obj_arr))
    res.title("fixed step")
    res.savefig("fixedstep.png")
