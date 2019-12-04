#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Gauss SVM
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size):
    a = np.linspace(0, 4 * np.pi, num=sample_size // 2)
    x = np.concatenate(
        [np.stack([a * np.cos(a), a * np.sin(a)], axis=1),
         np.stack([(a + np.pi) * np.cos(a), (a + np.pi) * np.sin(a)], axis=1)])
    x += np.random.random(size=x.shape)
    y = np.concatenate([np.ones(sample_size // 2), -np.ones(sample_size // 2)])
    return x, y


def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))


def optimize_param(design_mat, y, regularizer, lr):
    """

    :param design_mat: K
    :param y:teacher data
    :param regularizer: C
    :param lr: learnign rate
    :return:
    """
    prev_theta = np.ones((design_mat.shape[0]))
    theta = np.zeros((design_mat.shape[0]))
    EPSILON = 0.1
    while ((theta - prev_theta).T @ (theta - prev_theta) > EPSILON):
        prev_theta = theta
        condition = 1 - theta.T @ design_mat
        condition = condition.T
        assert condition.shape == theta.shape, str(condition.shape) + str(theta.shape)
        subgrad = - y @ design_mat
        assert subgrad.shape[0] == theta.shape[0], subgrad.shape
        index = condition < 0
        index = np.squeeze(index)
        subgrad[index] = 0
        update = regularizer * subgrad
        update += 2 * design_mat @ theta
        theta = theta - lr * update
    return theta

def visualize(theta, x, y, grid_size=100, x_min=-16, x_max=16):
    grid = np.linspace(x_min, x_max, grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
    design_mat = build_design_mat(x, mesh_grid, bandwidth=1.)
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.contourf(X, Y, np.reshape(np.sign(design_mat.T.dot(theta)),
                                  (grid_size, grid_size)),
                 alpha=.4, cmap=plt.cm.coolwarm)
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='$O$', c='blue')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='x', c='red')
    plt.savefig('ML1-homework2.png')


x, y = generate_data(sample_size=200)
design_mat = build_design_mat(x, x, bandwidth=1.)
theta = optimize_param(design_mat, y, regularizer=1, lr=0.0001)
visualize(theta, x, y)
