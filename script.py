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
    return (1 - x ** 3) * np.sin(2 * np.pi * x)

def get_poly_interpolant(X, Y):
    # Returns a polynomial interpolant for data points (X, Y)
    # TODO: improve documentation 
    if len(X) != len(Y) :
        raise ValueError("X and Y must have the same length.")
    n = len(X) - 1
    V = [[x ** i for i in range(n+1)] for x in X] 
    V = np.matrix(V) # Vandermonde matrix
    Y = np.transpose(np.matrix(Y))
    A = V.I.dot(Y) # Coefficients for polynomial 
    def interpolant(w): 
        w_exps = np.matrix([w ** i for i in range(n+1)]) # [1, w, w ** 2, ...]
        return w_exps.dot(A)[0, 0]
    return interpolant

X = np.linspace(0, 2, num = 201)
G = g(X)
#F = f(X)
plt.plot(X, G)
#plt.plot(X, F)
breaks = np.linspace(0, 2, num = 5)
p = get_poly_interpolant(breaks, g(breaks))
P = [p(x) for x in X] # TODO: Some sort of bug within the automatic vectorization?
plt.plot(X, P)

breaks = np.linspace(0, 2, num = 10)
p = get_poly_interpolant(breaks, g(breaks))
P = [p(x) for x in X] # TODO: This plotting thing should just be a function
plt.plot(X, P)

breaks = np.linspace(0, 2, num = 15)
p = get_poly_interpolant(breaks, g(breaks))
P = [p(x) for x in X] # TODO: Why is this so awful (and so good in the middle)
plt.plot(X, P)

plt.show()