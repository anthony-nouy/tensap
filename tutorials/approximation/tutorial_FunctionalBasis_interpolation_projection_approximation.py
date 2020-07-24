'''
Tutorial on FunctionalBasis, projection, interpolation, least-squares
approximation.

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

import numpy as np
import matplotlib.pyplot as plt
import tensap

# %% Interpolation on a polynomial space using Chebyshev Points
def FUN(x):
    # Function to approximate
    return np.cos(10*x)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(-1, 1)

# Interpolation basis and points
P = 30
H = tensap.PolynomialFunctionalBasis(X.orthonormal_polynomials(), range(P+1))
X_CP = tensap.chebyshev_points(H.cardinal(), [-1, 1])

# Interpolation of the function
F = H.interpolate(FUN, X_CP)

# Displays and error
print('Interpolation on a polynomial space using Chebyshev Points')
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(('True function', 'Interpolation'))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print('Mean squared error = %2.5e' % ERR_L2)

# %% Interpolation on a polynomial space using magic Points
def FUN(x):
    # Function to approximate
    return np.cos(10*x)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(-1, 1)

# Interpolation basis and points
P = 30
H = tensap.PolynomialFunctionalBasis(X.orthonormal_polynomials(), range(P+1))
X_MP = H.magic_points(X.random(10000))[0]

# Interpolation of the function
F = H.interpolate(FUN, X_MP)

# Displays and error
print('\nInterpolation on a polynomial space using magic points')
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(('True function', 'Interpolation'))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print('Mean squared error = %2.5e' % ERR_L2)

# %% Projection on polynomial space through quadrature
def FUN(x):
    # Function to approximate
    return x**2/2


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.NormalRandomVariable()

# Integration rule
I = X.gauss_integration_rule(5)

# Approximation basis
P = 3
H = tensap.PolynomialFunctionalBasis(X.orthonormal_polynomials(), range(P+1))

# Computation of the projection
F = H.projection(FUN, I)

# Displays and error
print('\nProjection on polynomial space through quadrature')
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print('Mean squared error = %2.5e' % ERR_L2)

# %% Interpolation on UserDefinedFunctionalBasis
def FUN(x):
    # Function to approximate
    return np.exp(-(x-1/2)**2)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(0, 1)

# Fourier Basis
N = 15
H = np.empty(2*N+1, dtype=object)
H[0] = lambda x: np.ones(np.size(x))
for i in np.arange(1, N+1):
    H[2*i-1] = lambda x, i=i: np.sqrt(2)*np.cos(2*np.pi*i*x)
    H[2*i] = lambda x, i=i: np.sqrt(2)*np.sin(2*np.pi*i*x)
H = tensap.UserDefinedFunctionalBasis(H, X)

# Display of the Fourier basis
plt.figure()
X_PLOT = np.linspace(0, 1, 100)
plt.plot(X_PLOT, H.eval(X_PLOT))
plt.title('Fourier basis')
plt.show()

# Computation of the interpolation
X_MP = H.magic_points(X.random(10000))[0]
F = H.interpolate(FUN, X_MP)

# Displays and error
print('\nInterpolation on UserDefinedFunctionalBasis')
plt.figure()
X_PLOT = np.linspace(0, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(('True function', 'Interpolation'))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print('Mean squared error = %2.5e' % ERR_L2)

# %% Interpolation with a radial basis
def FUN(x):
    # Function to approximate
    return np.exp(-x**2)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(-1, 1)

# Radial basis
N = 20
X_LIN = np.linspace(-1, 1, N)
S = 10/N
k = lambda x, y, s=S: np.exp(-(x-y)**2/s**2)
H = np.empty(N, dtype=object)
for i in range(N):
    H[i] = lambda y, x=X_LIN[i], k=k: k(y, x)
H = tensap.UserDefinedFunctionalBasis(H, X)

# Computation of the interpolation
X_MP = H.magic_points(X.random(100000))[0]
F = H.interpolate(FUN, X_MP)

# Displays and error
print('\nInterpolation with a radial basis')
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(('True function', 'Interpolation'))
plt.show()
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print('Mean squared error = %2.5e' % ERR_L2)

# %% Projection on polynomial space through quadrature
def FUN(x):
    # Function to approximate
    return x**2/2


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.NormalRandomVariable()

# Integration rule
I = X.gauss_integration_rule(5)

# Approximation basis
P = 3
H = tensap.PolynomialFunctionalBasis(tensap.HermitePolynomials(), range(P+1))

# Computation of the approximation
F = H.projection(FUN, I)

# Derivative and second derivative of F through projection
DF = H.projection(lambda x, F=F: F.eval_derivative(1, x), I)
DDF = H.projection(lambda x, DF=DF: DF.eval_derivative(1, x), I)

# Displays and error
print('\nProjection on polynomial space through quadrature')
N = 100
ERR_L2, ERR_L_INF = F.test_error(FUN, N, X)
print('Mean squared error = %2.5e' % ERR_L2)
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, F(X_PLOT))
plt.plot(X_PLOT, DF(X_PLOT))
plt.plot(X_PLOT, DDF(X_PLOT))
plt.legend(('F', 'df', 'ddf'))
plt.show()

# %% Least-squares approximation
def FUN(x):
    # Function to approximate
    return np.exp(-(x-1/2)**2)


FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True
X = tensap.UniformRandomVariable(0, 1)

# Approximation basis: Hermite polynomials of maximal degree P
P = 10
H = tensap.PolynomialFunctionalBasis(tensap.HermitePolynomials(), range(P+1))

# Solver
SOLVER = tensap.LinearModelLearningSquareLoss()
SOLVER.regularization = False
SOLVER.basis_adaptation = False

# Training sample
X_TRAIN = X.random(100)
Y_TRAIN = FUN(X_TRAIN)

# Computation of the approximation
SOLVER.basis = H
SOLVER.training_data = [X_TRAIN, Y_TRAIN]
F, OUTPUT = SOLVER.solve()

# Displays and error
print('\nLeast-squares approximation')
plt.figure()
X_PLOT = np.linspace(-1, 1, 100)
plt.plot(X_PLOT, FUN(X_PLOT))
plt.plot(X_PLOT, F(X_PLOT))
plt.legend(('True function', 'Interpolation'))
plt.show()
X_TEST = X.random(100)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print('Mean squared error = %2.5e' %
      (np.linalg.norm(Y_TEST-F_X_TEST) / np.linalg.norm(Y_TEST)))
