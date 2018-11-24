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
        # List of the form [0, 0, ..., 0, 1], with n 0's, which represents T_n
    roots = cheb.chebroots(c)
    def scale (r):
        return (r * (stop - start) / 2) + ((start + stop) / 2)
    roots = scale(roots)
    return roots

def B(x, k, a, h):
    # Cubic B-Spline Basis Function
    return spline_basis(x - (k * h), a, h)

def get_F(Y, h):
    # This comes from sec 3.5.3.2 in the module notes. 
    n = len(Y) - 1
    F = [Y[0] / (h ** 3),]
    for y in Y[1:n]:
        F = F + [y * 6 / (h ** 3),]
    F = F + [Y[-1] / (h ** 3),]
    return np.transpose(np.matrix(F))

def get_M(n):
    # This comes from identity 3.10 in the module notes. 
    M = [[1,] + [0 for i in range(n)],]
    for i in range(1, n):
        M = M + [[0 for i in range(i-1)] + [1, 4, 1] + [0 for i in range(n-i-1)]]
    M = M + [[0 for i in range(n)] + [1,]]
    return np.matrix(M)

def spline_basis(x, a, h):
    # This comes from section 3.5.3.2 in the module notes. 
    if x - a < -2 * h: 
        return 0
    if x - a < -h: 
        return (1 / 6) * ((2 * h + (x - a)) ** 3)
    if x - a < 0:
        return (2 * (h ** 3) / 3) - (1 / 2) * ((x - a) ** 2) * (2 * h + (x - a))
    if x - a < h:
        return (2 * (h ** 3) / 3) - (1 / 2) * ((x - a) ** 2) * (2 * h - (x - a))
    if x - a < 2 * h: 
        return (1 / 6) * ((2 * h - (x - a)) ** 3)
    return 0

def get_spline_interpolant(X, Y):
    if len(X) != len(Y) :
        raise ValueError("X and Y must have the same length.")
    n = len(X) - 1
    a = X[0]
    b = X[-1]
    h = (b - a) / n
    #print("n: {} a: {} b: {} h: {}".format(n, a, b, h))
    # The following come from identity 3.10 and sec 3.5.3.2 in the module notes. 
    F = get_F(Y, h)
    M = get_M(n)
    A = M.I.dot(F)
    A = [A[k, 0] for k in range(n+1)]
    A = [2 * A[0] - A[1]] + A + [2 * A[-1] + A[-2]]
        # Now A is a list with A[i] = a_i-1 where a_i-1 is the i-1'th coeffnt.
    def interpolant(w):
        s = 0
        for k in range(-1, n+2):
            s = s + A[k + 1] * B(w, k, a, h)
        #terms = [ lambda x: A[k + 1] * B( x, k, a, h ) for k in range(-1, n+2)] 
             # List of weighted basis functions, which are functions of x
        #s = sum([func(w) for func in terms]) 
            # Sum of weighted basis functions, that is, the interpolant
        return s
    return interpolant

def get_Y(X, func, method, breaks_method, n):
    # Gets Y's for plotting via various methods. 
    start = X[0]
    stop = X[-1]
    if breaks_method == "even":
        breaks = get_breaks_even(start, stop, n + 1) 
            # For a degree n polynomial we need n + 1 interpolation points. 
    if breaks_method == "cheb":
        breaks = get_breaks_cheb(start, stop, n + 1)
    if method == "poly":
        interpolant = get_poly_interpolant(breaks, func(breaks))
    if method == "spline":
        #global debug_X 
        #debug_X = breaks
        #global debug_Y
        #debug_Y = func(breaks)
        #print("X = {}".format(debug_X))
        #print("Y = {}".format(debug_Y))
        interpolant = get_spline_interpolant(breaks, func(breaks))
    Y = list(map(interpolant, X))
    return Y

X = np.linspace(0, 2, num = 201) # x's for plotting

# Question 1.1 
print("Question 1.1")
plt.plot(X, g(X))
print("blue: \tg")

Y = get_Y(X, g, "poly", "even", 5)
plt.plot(X, Y)
print("yellow:\tp with degree 5, and 6 evenly spaced interp points")

Y = get_Y(X, g, "poly", "even", 10)
plt.plot(X, Y)
print("green:\tp with degree 10, and 11 evenly spaced interp points")

plt.show()

# Question 1.2 
print("Question 1.2")
plt.plot(X, g(X))
print("blue: \tg")

Y = get_Y(X, g, "poly", "cheb", 5)
plt.plot(X, Y)
print("yellow:\tp with degree 5, and 6 Chebyshev interp points")

Y = get_Y(X, g, "poly", "cheb", 10)
plt.plot(X, Y)
print("green:\tp with degree 10, and 11 Chebyshev interp points")

plt.show()

# Question 1.3
debug_X, debug_Y = 0, 0

print("Question 1.3")
plt.plot(X, g(X))
print("blue: \tg")

Y = get_Y(X, g, "spline", "even", 5)
plt.plot(X, Y)
print("yellow:\tCubic spline from 6 evenly spaced interp points")

Y = get_Y(X, g, "spline", "even", 10)
plt.plot(X, Y)
print("green:\tCubic spline from 11 evenly spaced interp points")

plt.show()

