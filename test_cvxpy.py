import numpy as np
import cvxpy as cp
import functools
n = 10
# np.random.seed(3)

a = np.random.randint(1, 10, size=n)
b = np.random.randint(1, 10, size=n)
c = np.random.randint(1, 10, size=n)
d = np.random.randint(1, 10, size=n)
e = np.random.randint(1, 10, size=n)
print(a)
print(b)
print(sum(c))

x = cp.Variable(shape=n, boolean=True)
y = cp.Variable(shape=n, nonneg=True)
t = cp.Variable(shape=1, nonneg=True)
# objective function 


objective = cp.Minimize(cp.sum(x@a) / cp.sum(x@b))
constraints = [cp.sum(x@b) >= 1]

# constraints.append(cp.multiply(cp.sum(x@a), t) >= 10*cp.sum(x))
# constraints.append(cp.sum(cp.multiply(x, c)) >= 30)
# value = 3
# constraints.append(cp.sum(b@y) >= cp.sum((a-value*b)@x))
# constraints.append(cp.sum(b@y) >= cp.sum((value*b-a)@x))
# constraints.append(cp.sum(b@x) >= 1)
# for i in range(n):
#     constraints.append(y[i] <= 100*x[i])
#     constraints.append(t-y[i] >= 0)
#     constraints.append(t-y[i] <= 100*(1-x[i]))


problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC, verbose=0, qcp=True, maximumSeconds=10)
assert problem.is_dqcp()
print("Optimal value: ", problem.value)
print("t: ", t.value)
print("x: ", x.value)
print("value: ", sum(x.value*a)/sum(x.value*b))
