import numpy as np
import time
import cvxpy as cp
n = 10

a = np.random.randint(1, 10, size=n)
b = np.random.randint(1, 10, size=n)
c = np.random.randint(1, 10, size=n)
d = np.random.randint(1, 10, size=n)
e = np.random.randint(1, 10, size=n)
f = np.random.randint(1, 10, size=n)
g = np.random.randint(1, 10, size=n)

x = cp.Variable(shape=n, boolean=True)

# objective function 
objective = cp.Maximize(cp.sum(cp.multiply(x,a)))

# constraints
constraints = []

constraints.append(cp.sum_largest(cp.hstack([
    cp.sum(cp.multiply(x, b)), 
    cp.sum(cp.multiply(x, c)), 
    cp.sum(cp.multiply(x, d))]), 1) <= 10)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n/2)
# constraints.append(cp.sum(cp.multiply(x,a)) <= n)
# constraints.append(cp.sum(cp.multiply(x,b)) <= n/2)
# constraints.append(cp.sum(cp.multiply(x,c)) >= n/3)
# constraints.append(cp.sum(cp.multiply(x,d)) >= n/4)
# constraints.append(cp.sum(cp.multiply(x,e)) <= n/3)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)


"""
For constraints like `x*a > 100 if x*a != 0`, we can treat x*a as semi-continuous:
`100*delta <= a*x <= infinity*delta, delta = 1 or 0`

"""


prob = cp.Problem(objective, constraints)
# solve model
prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=100, numberThreads=4)
print("status:", prob.status)
print(sum(x.value)/n)

