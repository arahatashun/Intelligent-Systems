#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def chi_squared(num):
    x = np.random.normal(size=num)
    y = np.random.normal(size=num)
    return x * x + y * y


def t_dist(num):
    x = np.random.normal(size=num)
    y = chi_squared(num)
    return x / np.sqrt(y / 2)


def central_limit(func, mu):
    n = 10000
    res = []
    for i in range(10000):
        x_hat = np.mean(func(n))
        res.append((x_hat - mu) / np.sqrt(n))
    sns.distplot(res)
    plt.title("Central Limit Theorem")
    mu_, std_ = norm.fit(res)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_, std_)
    plt.plot(x, p, 'k', linewidth=2, label="Normal Distribution Fitting")
    plt.legend()
    plt.savefig(func.__name__ + "central.png")
    plt.show()

def strong_law(func):
    res = []
    for i in range(1, 10000):
        res.append(np.mean(func(i)))
    plt.plot(np.arange(1, 10000), res, label=func.__name__)
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\bar{X}_n$")
    plt.title("Strong Law of Large Numbers")
    plt.legend()
    plt.savefig(func.__name__ + "strong.png")
    plt.show()

if __name__ == '__main__':
    strong_law(chi_squared)
    central_limit(chi_squared, mu=2)
    strong_law(t_dist)
    central_limit(t_dist, mu=0)
