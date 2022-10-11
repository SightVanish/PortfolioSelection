import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize, LpBinary, LpStatus, value, PULP_CBC_CMD

save_path = "./demo_result.csv" # csv path
TIMEOUT = 10 # time out

NbvCost = True  # True: max nbv; False: min cost
minTotalNbv = 40000000
maxTotalNbv = 100000000
minTotalCost = 20000000
maxTotalCost = 100000000
fleetAgeLowBound = [None, 3, 4]
fleetAgeUpBound = [3, 6, 10]
fleetAgeLimit = [0.2, 0.3, 0.3]
fleetAgeGeq = [True, True, True] # True: >=, False: <=
OnHireLimit = 1.0
weightedAgeLowBound = [None, 3, 4]
weightedAgeUpBound = [4, 6, 15]
weightedAgeLimit = [0.4, 0.2, 0.3]
weightedAgeGeq = [True, True, True]
productType = ['D20', 'D4H', 'R4H']
productLimit = [0.1, 0.6, 0.0]
productGeq = [True, True, True]
lesseeType = ['MSC', 'ONE', 'HAPAG']
lesseeLimit = [0.5, 0.3, 0.2]
lesseeGeq = [False, False, False]
contractType = ['LT', 'LE', 'LF']
contractLimit = [0.3, 0.2, 0.15]
contractGeq = [True, True, True]

print('Data loading...')
n = 30000
rawData = pd.read_excel(io='./FCI ANZ (2022-07-08) (NBV as at 30 Jun 2022)_v2.xlsx', sheet_name='Raw (portfolio)', engine='openpyxl')
data = rawData.iloc[:n, :61]
print('Data processing...')
def SelectFleetAge(age, i):
    if fleetAgeLowBound[i] is None:
        fleetAgeLowBound[i] = -float('inf')
    if fleetAgeUpBound[i] is None:
        fleetAgeUpBound[i] = float('inf')
    if fleetAgeLowBound[i] <= age < fleetAgeUpBound[i]:
        age = 1
    else:
        age = 0
    return age
data['FleetAge1'] = data.apply(lambda x: SelectFleetAge(x['Fleet Year Fz'], 0), axis=1)
data['FleetAge2'] = data.apply(lambda x: SelectFleetAge(x['Fleet Year Fz'], 1), axis=1)
data['FleetAge3'] = data.apply(lambda x: SelectFleetAge(x['Fleet Year Fz'], 2), axis=1)

def SelectStatus(status):
    if status == 'ON':
        status = 1
    else:
        status = 0
    return status
data['Status'] = data.apply(lambda x: SelectStatus(x['Billing Status Fz']), axis=1)

def SelectWeightedAge(age, i):
    if weightedAgeLowBound[i] is None:
        weightedAgeLowBound[i] = -float('inf')
    if weightedAgeUpBound[i] is None:
        weightedAgeUpBound[i] = float('inf')
    if weightedAgeLowBound[i] <= age < weightedAgeUpBound[i]:
        age = 1
    else:
        age = 0
    return age
data['WeightedAge1'] = data.apply(lambda x: SelectWeightedAge(x['Age x CEU'], 0), axis=1)
data['WeightedAge2'] = data.apply(lambda x: SelectWeightedAge(x['Age x CEU'], 1), axis=1)
data['WeightedAge3'] = data.apply(lambda x: SelectWeightedAge(x['Age x CEU'], 2), axis=1)

def SelectProductType(product, i):
    if product == productType[i]:
        product = 1
    else:
        product = 0
    return product
data['ProductType1'] = data.apply(lambda x: SelectProductType(x['Product'], 0), axis=1)
data['ProductType2'] = data.apply(lambda x: SelectProductType(x['Product'], 1), axis=1)
data['ProductType3'] = data.apply(lambda x: SelectProductType(x['Product'], 2), axis=1)

def SelectLessee(lessee, i):
    if lessee == lesseeType[i]:
        lessee = 1
    else:
        lessee = 0
    return lessee
data['Lessee1'] = data.apply(lambda x: SelectLessee(x['Contract Cust Id'], 0), axis=1)
data['Lessee2'] = data.apply(lambda x: SelectLessee(x['Contract Cust Id'], 1), axis=1)
data['Lessee3'] = data.apply(lambda x: SelectLessee(x['Contract Cust Id'], 2), axis=1)

def SelectContractType(contract, i):
    if contract == contractType[i]:
        contract = 1
    else:
        contract = 0
    return contract
data['ContractType1'] = data.apply(lambda x: SelectContractType(x['Contract Lease Type'], 0), axis=1)
data['ContractType2'] = data.apply(lambda x: SelectContractType(x['Contract Lease Type'], 1), axis=1)
data['ContractType3'] = data.apply(lambda x: SelectContractType(x['Contract Lease Type'], 2), axis=1)

nbv = data['Nbv'].to_numpy()
cost = data['Cost'].to_numpy()
fleetAge = np.stack([data['FleetAge1'].to_numpy(),
                     data['FleetAge2'].to_numpy(),
                     data['FleetAge3'].to_numpy()])
status = data['Status'].to_numpy()
weightedAge = np.stack([data['WeightedAge1'].to_numpy(),
                        data['WeightedAge2'].to_numpy(),
                        data['WeightedAge3'].to_numpy()])
product = np.stack([data['ProductType1'].to_numpy(),
                        data['ProductType2'].to_numpy(),
                        data['ProductType3'].to_numpy()])
lessee = np.stack([data['Lessee1'].to_numpy(),
                   data['Lessee2'].to_numpy(),
                   data['Lessee3'].to_numpy()])
contract = np.stack([data['ContractType1'].to_numpy(),
                         data['ContractType2'].to_numpy(),
                         data['ContractType3'].to_numpy(),])

# Model
var = np.array([LpVariable('container_{0}'.format(i), lowBound=0, cat=LpBinary) for i in range(n)])
prob = LpProblem("MyProblem", LpMaximize if NbvCost else LpMinimize)

if NbvCost:
    prob += lpSum(var * nbv)
else:
    prob += lpSum(var * cost)

numSelected = lpSum(var)
if maxTotalNbv is not None:
    prob += lpSum(var * nbv) <= maxTotalNbv
if minTotalNbv is not None:
    prob += lpSum(var * nbv) >= minTotalNbv
if maxTotalCost is not None:
    prob += lpSum(var * cost) <= maxTotalCost
if minTotalCost is not None:
    prob += lpSum(var * cost) >= minTotalCost
if fleetAgeLimit[0] is not None:
    if fleetAgeGeq[0]:
        prob += lpSum(var * fleetAge[0]) >= fleetAgeLimit[0] * numSelected
    else:
        prob += lpSum(var * fleetAge[0]) <= fleetAgeLimit[0] * numSelected
if fleetAgeLimit[1] is not None:
    if fleetAgeGeq[1]:
        prob += lpSum(var * fleetAge[1]) >= fleetAgeLimit[1] * numSelected
    else:
        prob += lpSum(var * fleetAge[1]) <= fleetAgeLimit[1] * numSelected
if fleetAgeLimit[2] is not None:
    if fleetAgeGeq[2]:
        prob += lpSum(var * fleetAge[2]) >= fleetAgeLimit[2] * numSelected
    else:
        prob += lpSum(var * fleetAge[2]) <= fleetAgeLimit[2] * numSelected
if OnHireLimit is not None:
    prob += lpSum(var * status) >= OnHireLimit * numSelected
if weightedAgeLimit[0] is not None:
    if weightedAgeGeq[0]:
        prob += lpSum(var * weightedAge[0]) >= weightedAgeLimit[0] * numSelected
    else:
        prob += lpSum(var * weightedAge[0]) <= weightedAgeLimit[0] * numSelected
if weightedAgeLimit[1] is not None:
    if weightedAgeGeq[1]:
        prob += lpSum(var * weightedAge[1]) >= weightedAgeLimit[1] * numSelected
    else:
        prob += lpSum(var * weightedAge[1]) <= weightedAgeLimit[1] * numSelected
if weightedAgeLimit[2] is not None:
    if weightedAgeGeq[2]:
        prob += lpSum(var * weightedAge[2]) >= weightedAgeLimit[2] * numSelected
    else:
        prob += lpSum(var * weightedAge[2]) <= weightedAgeLimit[2] * numSelected
if productLimit[0] is not None:
    if productGeq[0]:
        prob += lpSum(var * product[0]) >= productLimit[0] * numSelected
    else:
        prob += lpSum(var * product[0]) <= productLimit[0] * numSelected
if productLimit[1] is not None:
    if productGeq[1]:
        prob += lpSum(var * product[1]) >= productLimit[1] * numSelected
    else:
        prob += lpSum(var * product[1]) <= productLimit[1] * numSelected
if productLimit[2] is not None:
    if productGeq[2]:
        prob += lpSum(var * product[2]) >= productLimit[2] * numSelected
    else:
        prob += lpSum(var * product[2]) <= productLimit[2] * numSelected
if lesseeLimit[0] is not None:
    if lesseeGeq[0]:
        prob += lpSum(var * lessee[0]) >= lesseeLimit[0] * numSelected
    else:
        prob += lpSum(var * lessee[0]) <= lesseeLimit[0] * numSelected
if lesseeLimit[1] is not None:
    if lesseeGeq[1]:
        prob += lpSum(var * lessee[1]) >= lesseeLimit[1] * numSelected
    else:
        prob += lpSum(var * lessee[1]) <= lesseeLimit[1] * numSelected
if lesseeLimit[2] is not None:
    if lesseeGeq[2]:
        prob += lpSum(var * lessee[2]) >= lesseeLimit[2] * numSelected
    else:
        prob += lpSum(var * lessee[2]) <= lesseeLimit[2] * numSelected
if contractLimit[0] is not None:
    if contractGeq[0]:
        prob += lpSum(var * contract[0]) >= contractLimit[0] * numSelected
    else:
        prob += lpSum(var * contract[0]) <= contractLimit[0] * numSelected
if contractLimit[1] is not None:
    if contractGeq[1]:
        prob += lpSum(var * contract[1]) >= contractLimit[1] * numSelected
    else:
        prob += lpSum(var * contract[1]) <= contractLimit[1] * numSelected
if contractLimit[2] is not None:
    if contractGeq[2]:
        prob += lpSum(var * contract[2]) >= contractLimit[2] * numSelected
    else:
        prob += lpSum(var * contract[2]) <= contractLimit[2] * numSelected

print('Model running...')
print('======================================================================')

solver = PULP_CBC_CMD(msg = True, timeLimit=TIMEOUT)
prob.solve(solver)

print('======================================================================')
print('Running result: ', LpStatus[prob.status])
print("target value: ", value(prob.objective))
# optimal -- optimal solution found (not mathmatical optimum)
# not solved -- no possible solution found (try to increase TIMEOUT)
# infeasible -- no solution exits

# if solution is found
if prob.status == 1 or prob.status == 2:
    result = np.array([var[i].varValue for i in range(n)])
    print(int(sum(result)), '/', n, 'containers are selected.')
    # save
    data.drop(['FleetAge1', 'FleetAge2', 'FleetAge3', 'Status', 'WeightedAge1', 'WeightedAge2', 'WeightedAge3', 'ProductType1', 'ProductType2', 'ProductType3', 'Lessee1', 'Lessee2', 'Lessee3', 'ContractType1', 'ContractType2', 'ContractType3'], axis=1, inplace=True)
    data.insert(loc=0, column="Selected", value=result)
    data.to_csv(save_path)
    print('Save result to', save_path)

    # debug
    if 1:
        print('======================================================================')
        print("nbv:", round(sum(result * nbv), 2), 'constraints:', minTotalNbv, maxTotalNbv)
        print("cost:", round(sum(result * cost), 2), 'constraints:', minTotalCost, maxTotalCost)

        print("container age: ")
        print("\t container age from {0} to {1}:".format(fleetAgeLowBound[0], fleetAgeUpBound[0]), round(sum(result * fleetAge[0])/sum(result), 2), 'constraints:', fleetAgeLimit[0])
        print("\t container age from {0} to {1}:".format(fleetAgeLowBound[1], fleetAgeUpBound[1]), round(sum(result * fleetAge[1])/sum(result), 2), 'constraints:', fleetAgeLimit[1])
        print("\t container age from {0} to {1}:".format(fleetAgeLowBound[2], fleetAgeUpBound[2]), round(sum(result * fleetAge[2])/sum(result), 2), 'constraints:', fleetAgeLimit[2])

        print('billing status:', sum(result * status)/sum(result), 'constraints:', OnHireLimit)

        print("weighted age: ")
        print("\t weighted age from {0} to {1}:".format(weightedAgeLowBound[0], weightedAgeUpBound[0]), round(sum(result * weightedAge[0])/sum(result), 2), 'constraints:', weightedAgeLimit[0])
        print("\t weighted age from {0} to {1}:".format(weightedAgeLowBound[1], weightedAgeUpBound[1]), round(sum(result * weightedAge[1])/sum(result), 2), 'constraints:', weightedAgeLimit[1])
        print("\t weighted age from {0} to {1}:".format(weightedAgeLowBound[2], weightedAgeUpBound[2]), round(sum(result * weightedAge[2])/sum(result), 2), 'constraints:', weightedAgeLimit[2])
        
        print("product: ")
        print("\t product {0}:".format(productType[0]), round(sum(result * product[0])/sum(result), 2), 'constraints:', productLimit[0])
        print("\t product {0}:".format(productType[1]), round(sum(result * product[1])/sum(result), 2), 'constraints:', productLimit[1])
        print("\t product {0}:".format(productType[2]), round(sum(result * product[2])/sum(result), 2), 'constraints:', productLimit[2])
        
        print("lessee: ")
        print("\t lessee {0}:".format(lesseeType[0]), round(sum(result * lessee[0])/sum(result), 2), 'constraints:', lesseeLimit[0])
        print("\t lessee {0}:".format(lesseeType[1]), round(sum(result * lessee[1])/sum(result), 2), 'constraints:', lesseeLimit[1])
        print("\t lessee {0}:".format(lesseeType[2]), round(sum(result * lessee[2])/sum(result), 2), 'constraints:', lesseeLimit[2])
        
        print("contract type: ")
        print("\t contract type {0}:".format(contractType[0]), round(sum(result * contract[0])/sum(result), 2), 'constraints:', contractLimit[0])
        print("\t contract type {0}:".format(contractType[1]), round(sum(result * contract[1])/sum(result), 2), 'constraints:', contractLimit[1])
        print("\t contract type {0}:".format(contractType[2]), round(sum(result * contract[2])/sum(result), 2), 'constraints:', contractLimit[2])
        