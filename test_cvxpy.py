import numpy as np
import cvxpy as cp
n = 10
np.random.seed(4)

a = np.random.randint(1, 10, size=n)
b = np.random.randint(1, 10, size=n)
c = np.random.randint(1, 10, size=n)
d = np.random.randint(1, 10, size=n)
e = np.random.randint(1, 10, size=n)
print(a)
# x = cp.Variable(shape=n, boolean=True)
# isMax = cp.Variable(shape=3, boolean=True)

# # constraints
# constraints = []
# constraints.append(cp.sum_largest(cp.hstack([
#     cp.sum(cp.multiply(x, b)), 
#     cp.sum(cp.multiply(x, c)),
#     cp.sum(cp.multiply(x, d))]), 2) <= 100)

# constraints.append(cp.sum(cp.multiply(x, e)) >= 10)

# # x*b
# constraints.append(cp.sum(cp.multiply(x, b)) >= cp.sum(cp.multiply(x, c)) - (1-isMax[0])*100)
# constraints.append(cp.sum(cp.multiply(x, b)) >= cp.sum(cp.multiply(x, d)) - (1-isMax[0])*100)

# # x*c
# constraints.append(cp.sum(cp.multiply(x, c)) >= cp.sum(cp.multiply(x, b)) - (1-isMax[1])*100)
# constraints.append(cp.sum(cp.multiply(x, c)) >= cp.sum(cp.multiply(x, d)) - (1-isMax[1])*100)

# # x*d
# constraints.append(cp.sum(cp.multiply(x, d)) >= cp.sum(cp.multiply(x, b)) - (1-isMax[2])*100)
# constraints.append(cp.sum(cp.multiply(x, d)) >= cp.sum(cp.multiply(x, c)) - (1-isMax[2])*100)

# constraints.append(cp.sum(isMax) >= 1)

# constraints.append(cp.sum(cp.multiply(x, c)) <= 30 + (100-30)*isMax[1])
# constraints.append(cp.sum(cp.multiply(x, d)) <= 30 + (100-30)*isMax[2])
# # b constraint
# # constraints.append(cp.sum(cp.multiply(x, b)) <= 50)
# constraints.append(cp.sum(cp.multiply(x, b)) <= 30 + (100-30)*isMax[0])


x = cp.Variable(shape=n, boolean=True)
z = cp.Variable(shape=1)
# objective function 
objective = cp.Minimize(z)
# objective = cp.Maximize(1)

constraints = []
constraints.append(cp.sum(cp.multiply(x, b)) <= 8)

constraints.append(z >= cp.sum(cp.multiply(x,a)) - 20)
constraints.append(z >= 20 - cp.sum(cp.multiply(x,a)))


prob = cp.Problem(objective, constraints)
# solve model
prob.solve(solver=cp.CBC, verbose=False, maximumSeconds=100, numberThreads=4)
print("status:", prob.status)
print('x: ', x.value)
if x.value is not None:
    print('target: ', sum(x.value*a))
    print('constraints: ', sum(x.value*b))
"""
Tips:
For constraints like `x*a > 100 if x*a != 0`, we can treat x*a as semi-continuous:
`100*delta <= a*x <= infinity*delta, delta = 1 or 0`
"""
