#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23 November 2018
Project on Polynomial Interpolation
MT3802 Numerical Analysis
Robson Edwards 
"""

import numpy as np, matplotlib.pyplot as plt

def g(x):
    return np.tan(np.sin(x ** 3))

def f(x):
    return (1 - x**3) * np.sin(2 * np.pi * x)

X = np.linspace(-1, 2, num = 301)
G = g(X)
F = f(X)
plt.plot(X, G)
plt.plot(X, F)
plt.show()