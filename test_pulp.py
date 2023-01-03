import random
from pulp import LpProblem, LpMaximize, lpSum, LpVariable, LpBinary
from pulp.apis import PULP_CBC_CMD
import time
import numpy as np
PROBLEM_SIZE = 80000
THREADS = 4
print('Prepare Model')
prob = LpProblem("KnapsackProblem", LpMaximize)
costs = np.array([random.randint(1, 1000) for i in range(PROBLEM_SIZE)])
print('cost done')
weights = np.array([[random.randint(1, 1000) for i in range(PROBLEM_SIZE)] for _ in range(100)])
print('weights done')
capacity = 1000 * 0.5 * 0.75 * PROBLEM_SIZE
# x = LpVariable.dicts("x", range(PROBLEM_SIZE), cat=LpBinary)
x = np.array([LpVariable('x_{0}'.format(i), cat=LpBinary) for i in range(PROBLEM_SIZE)])

print('x done')
prob += lpSum(x * costs)
for j in range(10):
    print(j, end=' ')
    prob += lpSum(x * weights[j]) <= capacity
print('')
start_time = time.time()
print('Solve Model')
prob.solve(solver=PULP_CBC_CMD(threads=THREADS, msg=True))

print('Time Cost:', time.time()-start_time)



