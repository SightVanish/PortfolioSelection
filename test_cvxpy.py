import numpy as np
import time
import cvxpy as cp
n = 20

a = np.random.randint(1, 10, size=n)
b = np.random.randint(1, 10, size=n)
c = np.random.randint(1, 10, size=n)
d = np.random.randint(1, 10, size=n)

x = cp.Variable(shape=n, boolean=True)

# objective function 
objective = cp.Maximize(1)

# constraints
constraints = []

constraints.append(\
    cp.sum_largest(cp.hstack([\
        cp.sum(cp.multiply(x, b)), \
        cp.sum(cp.multiply(x, c)), \
        cp.sum(cp.multiply(x, d))]), 1) <= 10)


prob = cp.Problem(objective, constraints)

# solve model
prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=100, numberThreads=4)
print("status:", prob.status)

