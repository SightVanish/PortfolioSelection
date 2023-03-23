import numpy as np
import cvxpy as cp
n = 10000
np.random.seed(3)
a = np.random.randint(1, 10, size=n)
b = np.random.randint(1, 10, size=n)

# print(a, b)

# Variables
x = cp.Variable(shape=n, boolean=True)
# Objective
objective = cp.Maximize(x@a)
# Constraints
constraints = []
constraints.append(x@a <= 3*cp.sum(x))
# Build & Solve Problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC, verbose=1, maximumSeconds=20, numberThreads=4)

# Print Result
print(problem.status)
print(problem.constants)
# print("x: ", x.value)
print("Optimal value: ", problem.value)