'''
Tutorial on multivariate functions, Tensor Grids, Projection.

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

import matplotlib.pyplot as plt
import numpy as np
import tensap

# %% Scalar-valued UserDefinedFunction
d = 3
fun = tensap.UserDefinedFunction('x0+x1+x2**4', d)
fun.evaluation_at_multiple_points = True
x = np.random.rand(4, d)
print('fun.eval(x) = \n%s\n' % fun.eval(x))
print('fun(x) =      \n%s\n' % fun(x))

# %% Vector-valued UserDefinedFunction
f = tensap.UserDefinedFunction('[x0, 100*x1]', d, 2)
f.evaluation_at_multiple_points = False
print('With f.evaluation_at_multiple_points == False, f(x) = \n%s\n' % f(x))
f.evaluation_at_multiple_points = True
print('With f.evaluation_at_multiple_points == True, f(x) =  \n%s\n' % f(x))

f = tensap.UserDefinedFunction('[x1,100*x2]', d, [1, 2])
f.evaluation_at_multiple_points = False
print('With f.evaluation_at_multiple_points == False, f(x) = \n%s\n' % f(x))
f.evaluation_at_multiple_points = True
print('With f.evaluation_at_multiple_points == True, f(x) =  \n%s\n' % f(x))

# %% Evaluation on a FullTensorGrid
g = tensap.FullTensorGrid(np.linspace(-1, 1, 50), d)
fx = fun.eval_on_tensor_grid(g)
g.plot_grid(marker='x')

# %% Evaluation on a SparseTensorGrid
I = tensap.MultiIndices.with_bounded_norm(d, 1, 5)
g = tensap.SparseTensorGrid(np.linspace(0, 1, 6), I, d)
fx = fun.eval_on_tensor_grid(g)
g.plot_grid(marker='x')

# %% Functional Bases
h = tensap.PolynomialFunctionalBasis(tensap.CanonicalPolynomials(), range(5))
H = tensap.FunctionalBases.duplicate(h, d)
grid = tensap.FullTensorGrid(np.linspace(-1, 1, 10), d)
x = grid.array()
Hx = H.eval(x)

# %% Sparse tensor functional basis
d = 2
p = 1
m = 4
h = tensap.PolynomialFunctionalBasis(tensap.CanonicalPolynomials(), range(5))
H = tensap.FunctionalBases.duplicate(h, d)
I = tensap.MultiIndices.with_bounded_norm(d, p, m)
Psi = tensap.SparseTensorProductFunctionalBasis(H, I)

print('Multi-indices: \n%s\n' % I.array)

finegrid = tensap.FullTensorGrid(np.arange(-1, 1.1, 0.1), d)
x = finegrid.array()
Psix = Psi.eval(x)

for i in range(I.cardinal()):
    finegrid.plot(Psix[:, i])
    plt.title(I.array[i, :])

# %% Projection on polynomial space through quadrature
d = 3
p = 3
fun = tensap.UserDefinedFunction('x0+x1**2+x2**3', d)
fun.evaluation_at_multiple_points = True
v = tensap.NormalRandomVariable()
X = tensap.RandomVector(v, d)
I = v.gauss_integration_rule(5).tensorize(d)
u = fun.eval_on_tensor_grid(I.points)

h = tensap.PolynomialFunctionalBasis(tensap.HermitePolynomials(), range(p+1))
H = tensap.FunctionalBases.duplicate(h, d)
H = tensap.FullTensorProductFunctionalBasis(H)

f, _ = H.projection(fun, I)

N_test = 100
x_test = X.random(N_test)
err_test = np.linalg.norm(f(x_test) - fun(x_test))/np.linalg.norm(fun(x_test))
print('Test error = %2.5e\n' % err_test)
