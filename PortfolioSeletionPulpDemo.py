import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize, LpBinary, LpStatus, value, PULP_CBC_CMD
import time
import heapq

save_path = './demo_result.csv'

TIMEOUT = 300 # timeout 

n = 200000 # number of containers

NbvCost = True  # True: max nbv; False: min cost

minTotalNbv = 100000000
maxTotalNbv = 200000000

minTotalCost = 100000000
maxTotalCost = 200000000

fleetAgeLowBound = [None, 3, None, None, None]
fleetAgeUpBound = [3, 6, None, None, None]
fleetAgeLimit = [0.7, 0.01, None, None, None]
fleetAgeGeq = [True, True, None, None, None] # True: >=, False: <=

fleetAgeAvgLimit = 3.0
fleetAgeAvgGeq = False

OnHireLimit = 0.85

weightedAgeLowBound = [3, 4, None, None, None]
weightedAgeUpBound = [6, 15, None, None, None]
weightedAgeLimit = [0.15, 0.01, None, None, None]
weightedAgeGeq = [True, True, None, None, None]

weightedAgeAvgLimit = 7.0
weightedAgeAvgGeq = False

productType = ['D4H', 'D20', 'D40', None, None]
productLimit = [0.5, 0.1, 0.01, None, None]
productGeq = [True, True, True, None, None]

lesseeType = ['MSC', 'ESSC', 'ONE', None, None]
lesseeLimit = [0.6, 0.2, 0.8, None, None]
lesseeGeq = [False, False, None, None, None]

topLesseeLimit = [0.6, 0.8, None]
topLesseeGeq = [False, False, None]

contractType = ['LT', 'LP', 'LM', None, None]
contractLimit = [0.6, 0.1, 0.01, None, None]
contractGeq = [True, True, True, None, None]

print('Data loading...')
data = pd.read_csv('./prepared_data_demo.csv')
data = data[:n]

start_time = time.time()
print('Data processing...')
def SelectFleetAge(age, i):
    if fleetAgeLowBound[i] is None:
        fleetAgeLowBound[i] = -float('inf')
    if fleetAgeUpBound[i] is None:
        fleetAgeUpBound[i] = float('inf')
    return 1 if fleetAgeLowBound[i]<=age<fleetAgeUpBound[i] else 0
for i in range(5):
    if fleetAgeLimit[i] is not None:
        column_name = 'FleetAge{0}'.format(i)
        data[column_name] = data.apply(lambda x: SelectFleetAge(x['Fleet Year Fz'], i), axis=1)

def SelectStatus(status):
    return 1 if status=='ON' else 0
data['Status'] = data.apply(lambda x: SelectStatus(x['Billing Status Fz']), axis=1)

def SelectWeightedAge(age, i):
    if weightedAgeLowBound[i] is None:
        weightedAgeLowBound[i] = -float('inf')
    if weightedAgeUpBound[i] is None:
        weightedAgeUpBound[i] = float('inf')
    return 1 if weightedAgeLowBound[i]<=age<weightedAgeUpBound[i] else 0
for i in range(5):
    if weightedAgeLimit[i] is not None:
        column_name = 'WeightedAge{0}'.format(i)
        data[column_name] = data.apply(lambda x: SelectWeightedAge(x['Age x CEU'], i), axis=1)

def SelectProductType(product, i):
    return 1 if product == productType[i] else 0
for i in range(5):
    if productLimit[i] is not None:
        column_name = 'ProductType{0}'.format(i)
        data[column_name] = data.apply(lambda x: SelectProductType(x['Product'], i), axis=1)

def OneHotLessee(lessee, name):
    return 1 if lessee==name else 0
for lesseeName in data['Contract Cust Id'].value_counts().index:
    data[lesseeName] = data.apply(lambda x: OneHotLessee(x['Contract Cust Id'], lesseeName), axis=1)

def SelectContractType(contract, i):
    if contract == contractType[i]:
        contract = 1
    else:
        contract = 0
    return contract
for i in range(5):
    if contractLimit[i] is not None:
        column_name = 'ContractType{0}'.format(i)
        data[column_name] = data.apply(lambda x: SelectContractType(x['Contract Lease Type'], i), axis=1)

nbv = data['Nbv'].to_numpy()
cost = data['Cost'].to_numpy()
status = data['Status'].to_numpy()
fleetAge = []
weightedAge = []
product = []
lessee = []
contract = []
for i in range(5):
    fleetAge.append(data['FleetAge{0}'.format(i)].to_numpy() if fleetAgeLimit[i] is not None else None)
    weightedAge.append(data['WeightedAge{0}'.format(i)].to_numpy() if weightedAgeLimit[i] is not None else None)
    product.append(data['ProductType{0}'.format(i)].to_numpy() if productLimit[i] is not None else None)
    lessee.append(data['Lessee{0}'.format(i)].to_numpy() if lesseeLimit[i] is not None else None)
    contract.append(data['ContractType{0}'.format(i)].to_numpy() if contractLimit[i] is not None else None)

fleetAgeAvg = data['Fleet Year Fz'].to_numpy()
weightedAgeAvg = data['Age x CEU'].to_numpy()

lesseeOneHot = {lesseeName: data[lesseeName].to_numpy() for lesseeName in data['Contract Cust Id'].value_counts().index}

print('Time consumed for data processing:', time.time()-start_time)
print('Model preparation...')

def SortTop(l, n):
    topN = heapq.nlargest(n, l, key=lambda x:x[1])
    return np.sum(np.stack([lesseeOneHot[topN[i][0]] for i in range(n)]), axis=0)

var = np.array([LpVariable('container_{0}'.format(i), lowBound=0, cat=LpBinary) for i in range(nbv.shape[0])])
prob = LpProblem("MyProblem", LpMaximize if NbvCost else LpMinimize)
# warm up
warmProb = LpProblem("WarmProblem", LpMaximize)
warmProb += lpSum(var * 1)
warmProb.solve(PULP_CBC_CMD(msg = False, timeLimit=1))

if NbvCost:
    prob += lpSum(var * nbv)
else:
    prob += lpSum(var * cost)

numSelected = lpSum(var) # num of selected containers

# nbv
if maxTotalNbv is not None:
    prob += lpSum(var * nbv) <= maxTotalNbv
if minTotalNbv is not None:
    prob += lpSum(var * nbv) >= minTotalNbv
    
# cost
if maxTotalCost is not None:
    prob += lpSum(var * cost) <= maxTotalCost
if minTotalCost is not None:
    prob += lpSum(var * cost) >= minTotalCost

# status
if OnHireLimit is not None:
    prob += lpSum(var * status) >= OnHireLimit * numSelected
    
# container age
if fleetAgeAvgLimit is not None:
    if fleetAgeAvgGeq:
        prob += lpSum(var * fleetAgeAvg) >= fleetAgeAvgLimit * numSelected
    else:
        prob += lpSum(var * fleetAgeAvg) <= fleetAgeAvgLimit * numSelected
for i in range(5):
    if fleetAgeLimit[i] is not None:
        if fleetAgeGeq[i]:
            prob += lpSum(var * fleetAge[i]) >= fleetAgeLimit[i] * numSelected
        else:
            prob += lpSum(var * fleetAge[i]) <= fleetAgeLimit[i] * numSelected

# weighted age
if weightedAgeAvgLimit is not None:
    if weightedAgeAvgGeq:
        prob += lpSum(var * weightedAgeAvg) >= weightedAgeAvgLimit * numSelected
    else:
        prob += lpSum(var * weightedAgeAvg) <= weightedAgeAvgLimit * numSelected
for i in range(5):
    if weightedAgeLimit[i] is not None:
        if weightedAgeGeq[i]:
            prob += lpSum(var * weightedAge[i]) >= weightedAgeLimit[i] * numSelected
        else:
            prob += lpSum(var * weightedAge[i]) <= weightedAgeLimit[i] * numSelected

# product
for i in range(5):
    if productLimit[i] is not None:
        if productGeq[i]:
            prob += lpSum(var * product[i]) >= productLimit[i] * numSelected
        else:
            prob += lpSum(var * product[i]) <= productLimit[i] * numSelected

# lessee
for i in range(5):
    if lesseeLimit[i] is not None:
        if lesseeGeq[i]:
            prob += lpSum(var * lesseeOneHot[lesseeType[i]]) >= lesseeLimit[i] * numSelected
        else:
            prob += lpSum(var * lesseeOneHot[lesseeType[i]]) <= lesseeLimit[i] * numSelected

# top1, 2, 3
for i in range(3):
    if topLesseeLimit[i] is not None:
        if topLesseeGeq[i]:
            prob += lpSum(var * SortTop(list({lesseeName: value(lpSum(var * lesseeOneHot[lesseeName])) for lesseeName in data['Contract Cust Id'].value_counts().index}.items()), i+1)) >= topLesseeLimit[i] * numSelected
        else:
            prob += lpSum(var * SortTop(list({lesseeName: value(lpSum(var * lesseeOneHot[lesseeName])) for lesseeName in data['Contract Cust Id'].value_counts().index}.items()), i+1)) <= topLesseeLimit[i] * numSelected

# contract type
for i in range(5):
    if contractLimit[i] is not None:
        if contractGeq[i]:
            prob += lpSum(var * contract[i]) >= contractLimit[i] * numSelected
        else:
            prob += lpSum(var * contract[i]) <= contractLimit[i] * numSelected

# prob.writeLP('problem.lp')

print('Model running...')
solver = PULP_CBC_CMD(msg = True, timeLimit=TIMEOUT)
prob.solve(solver)

print('Model Done.')
print("==============================================================")
# print(prob)
print("status:", LpStatus[prob.status])
print("==============================================================")
print("target value: ", value(prob.objective))
print('Total Time Consumed:', time.time()-start_time)
if 1:
    result = np.array([var[i].varValue for i in range(n)])
    print(int(sum(result)), '/', n, 'containers are selected.')
    print('======================================================================')
    print("nbv: {0} between {1} - {2}".format(round(sum(result * nbv), 2), minTotalNbv, maxTotalNbv))
    print("cost: {0} between {1} - {2}".format(round(sum(result * cost), 2), minTotalCost, maxTotalCost))
    print('billing status: {0}, -- {1}'.format(sum(result * status)/sum(result), OnHireLimit))

    print("container age: ")
    print('\t container average age is {0}, -- {1}'.format(round(sum(result * fleetAgeAvg)/sum(result), 2), fleetAgeAvgLimit))
    for i in range(5):
        if fleetAgeLimit[i] is not None:
            print("\t container age from {0} to {1} is {2}, -- {3}:".format(fleetAgeLowBound[i], fleetAgeUpBound[i], round(sum(result * fleetAge[i])/sum(result), 2), fleetAgeLimit[i]))

    print("weighted age: ")
    print('\t weighted average age is {0}, -- {1}'.format(round(sum(result * weightedAgeAvg)/sum(result), 2), weightedAgeAvgLimit))
    for i in range(5):
        if weightedAgeLimit[i] is not None:
            print("\t weighted age from {0} to {1} is {2}, -- {3}:".format(weightedAgeLowBound[i], weightedAgeUpBound[i], round(sum(result * weightedAge[i])/sum(result), 2), weightedAgeLimit[i]))    

    print("product: ")
    for i in range(5):
        if productLimit[i] is not None:
            print("\t product {0} is {1}, -- {2}:".format(productType[i], round(sum(result * product[i])/sum(result), 2), productLimit[i]))    
    
    print("lessee: ")
    for i in range(5):
        if lesseeLimit[i] is not None:
            print("\t lessee {0} is {1}, -- {2}:".format(lesseeType[i], round(sum(result * lessee[i])/sum(result), 2), lesseeLimit[i]))    

    print('Top lessee:')
    numLessee = {lesseeName: value(lpSum(var * lesseeOneHot[lesseeName])) for lesseeName in data['Contract Cust Id'].value_counts().index}
    sortedLessee = list(numLessee.items())
    top3Lessee = heapq.nlargest(3, sortedLessee, key=lambda x:x[1])
    print('\t top 1 {0} is {1}, -- {2}'.format(top3Lessee[0][0], top3Lessee[0][1] / sum(result), topLesseeLimit[0]))
    print('\t top 2 {0} {1} is {2}, -- {3}'.format(top3Lessee[0][0], top3Lessee[1][0], (top3Lessee[0][1] + top3Lessee[1][1])/ sum(result), topLesseeLimit[1]))

    
    print("contract type: ")
    for i in range(5):
        if contractLimit[i] is not None:
            print("\t contract type {0} is {1}, -- {2}:".format(contractType[i], round(sum(result * contract[i])/sum(result), 2), contractLimit[i])) 



if prob.status == 1 or prob.status == 2:
    print('Writing data...')
    data = data[['Unit Id Fz', 'Cost', 'Product', 'Contract Cust Id', 'Contract Lease Type', 'Nbv', 'Billing Status Fz', 'Fleet Year Fz', 'Age x CEU']]
    data.insert(loc=0, column="Selected", value=result)
    data.to_csv(save_path)
