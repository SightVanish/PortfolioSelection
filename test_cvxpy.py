import numpy as np
import cvxpy as cp
import functools
n = 10
np.random.seed(3)

a = np.random.randint(1, 10, size=n)
b = np.random.randint(1, 10, size=n)
c = np.random.randint(1, 10, size=n)
d = np.random.randint(1, 10, size=n)
e = np.random.randint(1, 10, size=n)
print(a)
print(b)
# print(sum(c))

x = cp.Variable(shape=n, boolean=True)
y = cp.Variable(shape=n, nonneg=True)
z = cp.Variable(shape=1, nonneg=True)
r = cp.Variable(shape=1, nonneg=True)

M = 10
# a*x/b*x -> 1.0
objective = cp.Minimize(cp.sum(x@a) / cp.sum(x@b))
constraints = []
constraints.append(cp.sum(x@b) >= 1)


problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CPLEX, verbose=0, qcp=1)
assert problem.is_dqcp()
print("Optimal value: ", problem.value)
print("ratio =", sum(x.value*a)/sum(x.value*b))

