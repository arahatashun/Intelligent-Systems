#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Backtracking line search and fixed step size comparison
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def optimize(xinit, func, grad, optimal_value, metric="fval", lineserch="armijo", L=None, iter=100, verbose=False):
    xs = []
    fvs = []
    grds = []
    x = xinit
    for i in range(iter):
        if verbose: print("iter:", i)
        xs.append(x)
        fv = func(x).item()
        d = grad(x)
        nd = LA.norm(d).item()
        if len(grds) == 0:
            grds.append(nd)
        else:
            grds.append(min(nd, grds[-1]))
        if metric == "fval":
            fvs.append(fv - optimal_value)
            if abs(fv - optimal_value) < EPS_CONVERGE:
                return [xs, fvs, grds]
        else:
            fvs.append(fv)
            if abs(nd - optimal_value) < EPS_CONVERGE:
                return [xs, fvs, grds]

        new_x = lambda e: x - e * d
        if lineserch == "armijo":
            x = x - backtrack(x, func, grad, new_x, d) * d
        else:
            x = x - 1.0 / L * d
    return [xs, fvs, grds]


def optimize_nestrov(xinit, func, grad, optimal_value, L, iter=1000, verbose=False):
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
        if verbose: print("iter:", i)
        xs.append(x)
        fv = func(x).item()
        fvs.append(fv - optimal_value)
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
    print("problems size: ", n)
    A = np.random.rand(m, n)
    winit = np.ones((n, 1))
    b = A @ winit + 0.1 * np.random.rand(m, 1)
    func = lambda w: (b - A @ w).T @ (b - A @ w)
    grad = lambda w: 2 * A.T @ (A @ w - b)
    wast = np.linalg.pinv(A) @ b
    optimal_value = func(wast).item()
    L = LA.norm(2 * A.T @ A, 'fro')
    return {"xinit": winit, "func": func, "grad": grad, "optimality": optimal_value, "L": L}


def strong_convex(m, n, lamd=1.0):
    """ minimize
    || b - Aw||_2^2 + lamd * ||w||_2^2
    :param m:
    :param n:
    :param lamd:
    :return:
    """
    print("problems size: ", n)
    A = np.random.rand(m, n)
    winit = np.ones((n, 1))
    b = A @ winit + 0.1 * np.random.rand(m, 1)
    func = lambda w: (b - A @ w).T @ (b - A @ w) + lamd * w.T @ w
    grad = lambda w: 2 * (A.T @ A + lamd * np.identity(n)) @ w - 2 * A.T @ b
    wast = np.linalg.solve(A.T @ A + lamd * np.identity(n), A.T @ b)
    optimal_value = func(wast).item()
    L = LA.norm(2 * (A.T @ A + lamd * np.identity(n)), 'fro')
    return {"xinit": winit, "func": func, "grad": grad, "optimality": optimal_value, "L": L}


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
    print("problems size: ", (n + 1) * l)
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
    return {"xinit": xinit, "func": func, "grad": grad, "optimality": 0}


def problem_one():
    fig = plt.figure(constrained_layout=False)
    # (b) convex
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title("(b) convex")
    # ax1.set_xlabel('iterations')
    cvx = convex(10, 100)
    xs, fvs, grds = optimize(cvx["xinit"], cvx["func"], cvx["grad"], cvx["optimality"], lineserch="constant",
                             L=cvx["L"], iter=1000)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.plot([i for i in range(len(fvs))], fvs, label="L-constant")
    # ax12 = ax1.twinx()
    # ax12.plot([i for i in range(len(fvs))], [1.0/v for v in fvs], label="L-constant")
    xs, fvs, grds = optimize(cvx["xinit"], cvx["func"], cvx["grad"], cvx["optimality"], lineserch="armijo",
                             L=cvx["L"], iter=1000)
    ax1.plot([i for i in range(len(fvs))], fvs, label="armijo")
    ax1.legend()
    # (d) convex +  acceleration
    ax2 = fig.add_subplot(3, 2, 2)
    # ax2.set_xlabel('iterations')
    ax2.set_title("(d) convex + acceleration")
    xs, fvs = optimize_nestrov(cvx["xinit"], cvx["func"], cvx["grad"], cvx["optimality"], L=cvx["L"], iter=1000)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.plot([i for i in range(len(fvs))], fvs, label="L-constant")
    # ax22 = ax2.twinx()
    # ax22.plot([i for i in range(len(fvs))], [pow(f, -2) for f in fvs], label="L-constant")
    ax2.legend()

    ax3 = fig.add_subplot(3, 2, 3)
    # ax3.set_xlabel('iterations')
    ax3.set_yscale('log')
    ax3.set_title("(c) strongly convex: $\lambda = 0.1$")
    stcvx = strong_convex(10, 100, lamd=0.1)
    xs, fvs, grds = optimize(stcvx["xinit"], stcvx["func"], stcvx["grad"], stcvx["optimality"], lineserch="constant",
                             L=stcvx["L"], iter=1000)
    ax3.get_xaxis().get_major_formatter().set_useOffset(False)
    ax3.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.plot([i for i in range(len(fvs))], fvs, label="L-constant")
    xs, fvs, grds = optimize(stcvx["xinit"], stcvx["func"], stcvx["grad"], stcvx["optimality"], lineserch="armijo",
                             L=stcvx["L"], iter=1000)
    ax3.plot([i for i in range(len(fvs))], fvs, label="armijo")
    ax3.legend()

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_yscale('log')
    # ax4.set_xlabel('iterations')
    ax4.set_title("(c) strongly convex: $\lambda = 10$")
    stcvx = strong_convex(10, 100, lamd=10)
    xs, fvs, grds = optimize(stcvx["xinit"], stcvx["func"], stcvx["grad"], stcvx["optimality"], lineserch="constant",
                             L=stcvx["L"], iter=1000)
    ax4.get_xaxis().get_major_formatter().set_useOffset(False)
    ax4.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.plot([i for i in range(len(fvs))], fvs, label="L-constant")
    xs, fvs, grds = optimize(stcvx["xinit"], stcvx["func"], stcvx["grad"], stcvx["optimality"], lineserch="armijo",
                             L=stcvx["L"], iter=1000)
    ax4.plot([i for i in range(len(fvs))], fvs, label="armijo")
    ax4.legend()

    ax5 = fig.add_subplot(3, 2, 5)
    # ax5.set_xlabel('iterations')
    ncvx = nonconvex(5, 4, 20)
    xs, fvs, grds = optimize(ncvx["xinit"], ncvx["func"], ncvx["grad"], ncvx["optimality"], metric="grad", iter=1000)
    ax5.set_title("(a) nonconvex")
    ax5.get_xaxis().get_major_formatter().set_useOffset(False)
    ax5.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax5.set_yscale('log')
    ax5.plot([i for i in range(len(fvs))], grds, label="armijo")
    ax5.legend()

    plt.tight_layout()
    plt.savefig("probem1.pdf")
    plt.show()


if __name__ == '__main__':
    problem_one()
