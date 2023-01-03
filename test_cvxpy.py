import numpy as np
import time
import cvxpy as cp
n = 20000

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

# constraints.append(\
#     cp.sum_largest(cp.hstack([\
#         cp.sum(cp.multiply(x, b)), \
#         cp.sum(cp.multiply(x, c)), \
#         cp.sum(cp.multiply(x, d))]), 3) <= n*2)
constraints.append(cp.sum(cp.multiply(x,a)) >= n/2)
constraints.append(cp.sum(cp.multiply(x,a)) <= n)
constraints.append(cp.sum(cp.multiply(x,b)) <= n/2)
constraints.append(cp.sum(cp.multiply(x,c)) >= n/3)
constraints.append(cp.sum(cp.multiply(x,d)) >= n/4)
constraints.append(cp.sum(cp.multiply(x,e)) <= n/3)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)
# constraints.append(cp.sum(cp.multiply(x,a)) >= n)


prob = cp.Problem(objective, constraints)

# solve model
prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=100, numberThreads=4)
print("status:", prob.status)
print(sum(x.value)/n)

