#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Backtracking line search and fixed step size comparison
"""
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility

ALPHA = 0.5
BETA = 0.8
EPSILON = 1.0
EPS_CONVERGE = 0.0001


def armijo(x, func, grad, new_x, epsilon):
    """ armijo condition
    :param x:
    :param d: direction lambda function
    :param epsilon:
    :return:
    """
    g = grad(x)
    left = func(new_x) - func(x)
    right = - ALPHA * epsilon * g.T @ (new_x - x)
    res = left.item() > right.item()
    # print("left:", left.item(), "right:", right.item(), "res:", res)
    assert type(res) is bool, type(res)
    return res


def backtrack(x, func, grad, new_x, d):
    eps = EPSILON
    while armijo(x, func, grad, new_x(eps), eps):
        eps = BETA * eps
    return eps


def optimize(xinit, func, grad, optimal_value, iter=100):
    xs = []
    fvs = []
    x = xinit
    for i in range(iter):
        print("iter:", i)
        xs.append(x)
        fv = func(x).item()
        fvs.append(fv)
        if abs(fv - optimal_value) < EPS_CONVERGE:
            return [xs, fvs]
        d = grad(x)
        new_x = lambda e: x - e * d
        x = x - backtrack(x, func, grad, new_x, d) * d
    return [xs, fvs]


def optimize_nestrov(xinit, func, grad, optimal_value, L, iter=1000):
    """ Nesterov's Accelerated Gradient Method

    :param xinit:
    :param func:
    :param grad:
    :param optimal_value:
    :param L: gradient lipshtiz constant
    :param iter:
    :return:
    """
    xs = []
    fvs = []
    x = xinit
    y = xinit
    alpha = 1.0 / L
    beta = lambda k: k / (k + 3)
    tau = 0
    for i in range(iter):
        print("iter:", i)
        xs.append(x)
        fv = func(x).item()
        fvs.append(fv)
        if abs(fv - optimal_value) < EPS_CONVERGE:
            return [xs, fvs]
        d = grad(x)
        new_x = y - alpha * d
        y = new_x + beta(i + 1) * (new_x - x)
        x = new_x
    return [xs, fvs]


def convex(m, n):
    """ minimize
    || b -A w||_2^2
    :param m:
    :param n:
    :return:
    """
    assert m < n, "set m < n"
    A = np.random.rand(m, n)
    winit = np.ones((n, 1))
    b = A @ winit + 0.1 * np.random.rand(m, 1)
    func = lambda w: (b - A @ w).T @ (b - A @ w)
    grad = lambda w: 2 * A.T @ (A @ w - b)
    wast = np.linalg.pinv(A) @ b
    optimal_value = func(wast)
    L = LA.norm(2 * A.T @ A, 'fro')
    xs, fvs = optimize_nestrov(winit, func, grad, optimal_value, L)
    plt.plot([i for i in range(len(fvs))], fvs)
    plt.show()


def strong_convex(m, n, lamd=1):
    """ minimize
    || b - Aw||_2^2 + lamd * ||w||_2^2
    :param m:
    :param n:
    :param lamd:
    :return:
    """
    A = np.random.rand(m, n)
    winit = np.ones((n, 1))
    b = A @ winit + 0.1 * np.random.rand(m, 1)
    func = lambda w: (b - A @ w).T @ (b - A @ w) + lamd * w.T @ w
    grad = lambda w: 2 * (A.T @ A + lamd * np.identity(n)) @ w - 2 * A.T @ b
    wast = np.linalg.solve(A.T @ A + lamd * np.identity(n), A.T @ b)
    optimal_value = func(wast)
    L = LA.norm(2 * (A.T @ A + lamd * np.identity(n)), 'fro')
    xs, fvs = optimize_nestrov(winit, func, grad, optimal_value, L)
    plt.plot([i for i in range(len(fvs))], fvs)
    plt.show()


def nonconvex(m, n, l):
    """ minimize
    ||b-AW1w2||_2^2
    W1 n times l
    w2 l
    :param m:
    :param n:
    :param l:
    :return:
    """
    A = np.random.rand(m, n)
    w2init = np.ones((l, 1))
    W1init = np.ones((n, l))
    b = A @ W1init @ w2init + 0.1 * np.random.rand(m, 1)
    funcbymat = lambda W1, w2: (b - A @ W1 @ w2).T @ (b - A @ W1 @ w2)
    gradW1 = lambda W1, w2: -2 * A.T @ (b - A @ W1 @ w2) @ w2.T
    gradw2 = lambda W1, w2: -2 * W1.T @ A.T @ (b - A @ W1 @ w2)
    # inorder to optimize
    func = lambda w: funcbymat(w[:n * l].reshape(n, l), w[n * l:].reshape(l, 1))
    grad = lambda w: np.concatenate([gradW1(w[:n * l].reshape(n, l), w[n * l:].reshape(l, 1)).flatten(),
                                    gradw2(w[:n * l].reshape(n, l), w[n * l:].reshape(l, 1)).flatten()])
    xinit = np.ones(n * l + l)
    xs, fvs = optimize(xinit, func, grad, 0, iter=100)
    plt.plot([i for i in range(len(fvs))], fvs)
    plt.show()
    
if __name__ == '__main__':
    nonconvex(20, 200, 10)
