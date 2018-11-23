#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23 November 2018
Project on Polynomial Interpolation
MT3802 Numerical Analysis
Robson Edwards 
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

def g(x):
    return np.tan(np.sin(x ** 3))

def f(x):
    return (1 - x ** 3) * np.sin(2 * np.pi * x)

def get_poly_interpolant(X, Y):
    # Returns a polynomial interpolant for diven data points
    # Args:
    #   X: list of x's (knots)
    #   Y: list of corresponding y's. That is, Y[i] = f(X[i]) for all i. 
    # Returns:
    #   a FUNCTION which is the polynomial interpolant of degree (len(X) - 1)
    
    if len(X) != len(Y) :
        raise ValueError("X and Y must have the same length.")
    n = len(X) - 1
    V = np.matrix ( [[x ** i for i in range(n+1)] for x in X] )# Vandermonde Mtx
    Y = np.transpose(np.matrix(Y))
    A = V.I.dot(Y) # Coefficients for polynomial 
    def interpolant(w): 
        w_exps = np.matrix([w ** i for i in range(n+1)]) # [1, w, w ** 2, ...]
        return w_exps.dot(A)[0, 0] # w_exps.dot(A) is a 1 x 1 matrix
    return interpolant

def get_breaks_even(start, stop, n):
    # Gets n evenly spaced break points between start and stop, inclusive. 
    return np.linspace(start, stop, n)

def get_breaks_cheb(start, stop, n):
    # Gets n break points between start and stop, according to the method for
    # minimising error bound by Chebyshev polynomials in module notes 3.4.1. 
    c = [0 for i in range(n)] + [1,] 
        # List of the form [0, 0, ..., 0, 1] which represents T_n
    roots = cheb.chebroots(c)
    def scale (r):
        return (r * (stop - start) / 2) + ((start + stop) / 2)
    roots = scale(roots)
    return roots

def get_Y(X, method, breaks_method, n):
    # Gets Y's for plotting via various methods. 
    start = X[0]
    stop = X[-1]
    if breaks_method == "even":
        breaks = get_breaks_even(start, stop, n + 1) 
            # For a degree n polynomial we need n + 1 interpolation points. 
    if breaks_method == "cheb":
        breaks = get_breaks_cheb(start, stop, n + 1)
    if method == "poly":
        p = get_poly_interpolant(breaks, g(breaks))
    if method == "splines":
        return
    P = list(map(p, X))
    return P

X = np.linspace(0, 2, num = 201) # x's for plotting

# Question 1.1 
plt.plot(X, g(X))
print("blue: \tg")

Y = get_Y(X, "poly", "even", 5)
plt.plot(X, Y)
print("yellow:\tp with degree 5, and 6 evenly spaced interp points")

Y = get_Y(X, "poly", "even", 10)
plt.plot(X, Y)
print("green:\tp with degree 10, and 11 evenly spaced interp points")

plt.show()

#Question 1.2 
plt.plot(X, g(X))
print("blue: \tg")

Y = get_Y(X, "poly", "cheb", 5)
plt.plot(X, Y)
print("green:\tp with degree 5, and 6 Chebyshev interp points")

Y = get_Y(X, "poly", "cheb", 10)
plt.plot(X, Y)
print("green:\tp with degree 10, and 11 Chebyshev interp points")

plt.show()