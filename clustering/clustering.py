#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
1. PCA dimension reduction (use eigh)
2. k-means
"""

import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

np.random.seed(0)  # set the random seed for reproducibility


def sampu_matrix(data):
    """ 散布行列を求める

    """
    C = data.T @ data
    """
    C = np.zeros((4, 4))
    for i in range(len(data)):
        x = data[i, :][:, None]
        print(x.shape)
        add = x @ x.T
        assert add.shape == (4, 4), add.shape
        C += add
    assert C.shape == (4, 4), C.shape
    """
    return C


def solve_pca(data):
    C = sampu_matrix(data)
    w, v = eigh(C)
    # w: eigenvalues in ascending order
    # v: The column v[:, i] is the normalized
    #     eigenvector corresponding to the eigenvalue w[i]
    # print("w: ", w)
    # print("v: ", v)
    T = np.array([v[:, 3], v[:, 2]])
    return T


def get_projected_data(data, pca):
    # print(data.shape)
    # print(pca.shape)
    return data @ pca.T


def kmeans(data):
    centers = np.array([data[0], data[1], data[2]]).T  # random initializati
    # print(centers.shape)
    EPSILON = 0.1
    diff = 1000
    n_data = data.shape[0]
    assign = np.zeros(n_data)
    while (diff > EPSILON):
        for i in range(n_data):
            dist = np.zeros(3)
            for j in range(3):
                dist[j] = np.linalg.norm(centers[:, j] - data[i, :])
            # print(dist)
            assign[i] = np.argmin(dist)
        # reassign center
        # print(assign)
        tmp_centers = np.random.rand(2, 3)
        cnt = np.zeros(3)
        for i in range(n_data):
            tmp_centers[:, int(assign[i])] += data[i, :]
            cnt[int(assign[i])] += 1
        for j in range(3):
            tmp_centers[:, j] = tmp_centers[:, j] / cnt[j]
        diff = 0
        for i in range(3):
            diff += np.linalg.norm(centers[:, i] - tmp_centers[:, i])
        centers = tmp_centers[:, :]
        # print("diff:", diff)
    return (centers, assign)


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    T = solve_pca(X)
    hidden = get_projected_data(X, T)
    fig = plt.figure(figsize=(12, 8))
    markers = ['o', '^', 'v']
    colors = ['b', 'g', 'r']
    for i in range(3):
        d = hidden[iris.target == i, :]
        plt.plot(d[:, 0], d[:, 1],'o', fillstyle='none', marker=markers[i], color=colors[i])
    plt.legend(iris.target_names)
    plt.title("PCA")
    #plt.show()
    plt.savefig("PCA.png")
    fig = plt.figure(figsize=(12, 8))
    center, assign = kmeans(hidden)
    for i in range(len(hidden)):
        d = hidden[i]
        plt.plot(d[0], d[1], 'o',fillstyle='none', marker=markers[int(assign[i])],
                 color=colors[int(assign[i])])
    plt.title("k-means")
    plt.savefig("k-means.png")
    #plt.show()


if __name__ == '__main__':
    main()
