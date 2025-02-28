import numpy as np
import tensap

# %% Gauss Integration Rule
# Gauss integration on [0,2]
mu = tensap.LebesgueMeasure(0, 2)
G = mu.gauss_integration_rule(10)
# Function f(x) = exp(x)
f = tensap.UserDefinedFunction("np.exp(x0)", 1)
Iapp = G.integrate(f)
Iex = np.exp(2) - 1
print("Error = %2.5e" % (np.abs(Iapp - Iex) / np.abs(Iex)))

# %% Piecewise Gauss-Legendre Integration Rule
g = tensap.IntegrationRule.gauss_legendre_composite([0, 0.5, 1], 6)
f = tensap.UserDefinedFunction("np.exp(x0)", 1)
Iapp = g.integrate(f)
Iex = np.exp(1) - 1

print("Integration error = %2.5e" % (np.abs(Iapp - Iex) / np.abs(Iex)))


# %% Tensor product integration rule
mu = tensap.ProductMeasure([tensap.LebesgueMeasure(0, 5), tensap.LebesgueMeasure(0, 1)])
G = mu.gauss_integration_rule([10, 2])
G = G.integration_rule()
# Function f(x0,x1) = exp(x0)*x1
f = tensap.UserDefinedFunction("np.exp(x0)*x1", 2)
f.evaluation_at_multiple_points = True
Iapp = G.integrate(f)
Iex = (np.exp(5) - 1) / 2
print("Error = %2.5e" % (np.abs(Iapp - Iex) / np.abs(Iex)))
