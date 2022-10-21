import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize, LpBinary, LpStatus, value, PulpSolverError, PULP_CBC_CMD
import pulp as pl
import time
import heapq
import json
import psycopg2
import psycopg2.extras
import requests
import warnings
import sys
import uuid
start_time = time.time()

TIMEOUT = 500 # timeout 

n = 100000 # number of containers
numLimit = 5

sqlData = "select billing_status_fz as billing, unit_id_fz, product, fleet_year_fz as fleet_year, contract_cust_id as customer, contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age from fll_t_dw.ads_fir_pkg_data LIMIT {0}".format(n) 

if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)

print("==============================================================")
print('Data loading...')

# load query id
url = ''
try:
    json_response = requests.get(url).content.decode()
    print(json_response)
except:
    print('http fails')

with open("./parameterDemo.json") as f:
    paramDict = json.load(f)

try:
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
    data = pd.read_sql(sqlData, conn)
    conn.close()
except:
    print("Loading data from GreenPlum failed!")

print('Time for loading data', time.time() - start_time)
mid_time = time.time()

queryID = paramDict['query_id']
initialQuery = paramDict['initial_query']
# TODO: link to database system
param = paramDict['secondary_query']
NbvCost = param['prefer']['NBVOrCost']
maxOrMin = param['prefer']['maxOrMin']

fleetAgeLowBound = [None for _ in range(numLimit)]
fleetAgeUpBound = [None for _ in range(numLimit)]
fleetAgeLimit = [None for _ in range(numLimit)]
fleetAgeGeq = [None for _ in range(numLimit)]
weightedAgeLowBound = [None for _ in range(numLimit)]
weightedAgeUpBound = [None for _ in range(numLimit)]
weightedAgeLimit = [None for _ in range(numLimit)]
weightedAgeGeq = [None for _ in range(numLimit)]
lesseeType = [None for _ in range(numLimit)]
lesseeLimit = [None for _ in range(numLimit)]
lesseeGeq = [None for _ in range(numLimit)]
productType = [None for _ in range(numLimit)]
productLimit = [None for _ in range(numLimit)]
productGeq = [None for _ in range(numLimit)]
contractType = [None for _ in range(numLimit)]
contractLimit = [None for _ in range(numLimit)]
contractGeq = [None for _ in range(numLimit)]

minTotalNbv = param['totalNBVFrom'] if param['totalNBVFrom'] else None
maxTotalNbv = param['totalNBVTo'] if param['totalNBVTo'] else None

minTotalCost = param['totalCostFrom'] if param['totalCostFrom'] else None
maxTotalCost = param['totalCostTo'] if param['totalCostTo'] else None

fleetAgeAvgLimit = param['containersAge']['average']['averageWeighedAge'] if param['containersAge']['average']['averageWeighedAge'] else None
fleetAgeAvgGeq = param['containersAge']['average']['symbol']
for i in range(len(param['containersAge']['list'])):
    fleetAgeLowBound[i] = param['containersAge']['list'][i]['containersAgeFrom']
    fleetAgeUpBound[i] = param['containersAge']['list'][i]['containersAgeTo']
    fleetAgeLimit[i] = param['containersAge']['list'][i]['%'] / 100
    fleetAgeGeq[i] = param['containersAge']['list'][i]['symbol']

weightedAgeAvgLimit = param['weightedAge']['average']['averageWeighedAge'] if param['weightedAge']['average']['averageWeighedAge'] else None
weightedAgeAvgGeq = param['weightedAge']['average']['symbol']
for i in range(len(param['weightedAge']['list'])):
    weightedAgeLowBound[i] = param['weightedAge']['list'][i]['weightedAgeFrom']
    weightedAgeUpBound[i] = param['weightedAge']['list'][i]['weightedAgeTo']
    weightedAgeLimit[i] = param['weightedAge']['list'][i]['%'] / 100
    weightedAgeGeq[i] = param['weightedAge']['list'][i]['symbol']

topLesseeLimit = [param['TopLessee'][0]['%'] / 100 if param['TopLessee'][0]['Top1'] else None,
               param['TopLessee'][1]['%'] / 100 if param['TopLessee'][1]['Top2'] else None,
               param['TopLessee'][2]['%'] / 100 if param['TopLessee'][2]['Top3'] else None]
topLesseeGeq = [param['TopLessee'][0]['symbol'] if param['TopLessee'][0]['Top1'] else None,
             param['TopLessee'][1]['symbol'] if param['TopLessee'][1]['Top2'] else None,
             param['TopLessee'][2]['symbol'] if param['TopLessee'][2]['Top3'] else None]
for i in range(len(param['lessee'])):
    lesseeType[i] = param['lessee'][i]['lessee']
    lesseeLimit[i] = param['lessee'][i]['%'] / 100
    lesseeGeq[i] = param['lessee'][i]['symbol']

OnHireLimit = param['status'][0]['%'] / 100 if param['status'][0]['onhire'] else None
OffHireLimit = param['status'][1]['%'] / 100 if param['status'][1]['offhire'] else None
NoneHireLimit = param['status'][2]['%'] / 100 if param['status'][2]['none'] else None

for i in range(len(param['product'])):
    productType[i] = param['product'][i]['productType']
    productLimit[i] = param['product'][i]['%'] / 100
    productGeq[i] = param['product'][i]['symbol']

for i in range(len(param['contractType'])):
    contractType[i] = param['contractType'][i]['contractType']
    contractLimit[i] = param['contractType'][i]['%'] / 100
    contractGeq[i] = param['contractType'][i]['symbol']

print("==============================================================")
print('Data processing...')
for i in range(5):
    if fleetAgeLimit[i]:
        column_name = 'FleetAge{0}'.format(i)
        data[column_name] = data['fleet_year'].apply(lambda x: 1 if fleetAgeLowBound[i]<=x<fleetAgeUpBound[i] else 0)

data['OnHireStatus'] = data['billing'].apply(lambda x: 1 if x=='ON' else 0)
data['OffHireStatus'] = data['billing'].apply(lambda x: 1 if x=='OF' else 0)
data['NoneStatus'] = data['billing'].apply(lambda x: 1 if x==' ' else 0)

for i in range(5):
    if weightedAgeLimit[i]:
        column_name = 'WeightedAge{0}'.format(i)
        data[column_name] = data['weighted_age'].apply(lambda x: 1 if weightedAgeLowBound[i]<=x<weightedAgeUpBound[i] else 0)

for i in range(5):
    if productLimit[i]:
        column_name = 'ProductType{0}'.format(i)
        data[column_name] = data['product'].apply(lambda x: 1 if x in productType[i] else 0)

for lesseeName in data['customer'].value_counts().index:
    data[lesseeName] = data['customer'].apply(lambda x: 1 if x==lesseeName else 0)

for i in range(5):
    if contractLimit[i]:
        column_name = 'ContractType{0}'.format(i)
        data[column_name] = data['contract'].apply(lambda x: 1 if x in contractType[i] else 0)

nbv = data['nbv'].to_numpy()
cost = data['cost'].to_numpy()
onHireStatus = data['OnHireStatus'].to_numpy()
offHireStatus = data['OffHireStatus'].to_numpy()
noneHireStatus = data['NoneStatus'].to_numpy()

fleetAge = []
weightedAge = []
product = []
contract = []
for i in range(numLimit):
    fleetAge.append(data['FleetAge{0}'.format(i)].to_numpy() if fleetAgeLimit[i] else None)
    weightedAge.append(data['WeightedAge{0}'.format(i)].to_numpy() if weightedAgeLimit[i] else None)
    product.append(data['ProductType{0}'.format(i)].to_numpy() if productLimit[i] else None)
    contract.append(data['ContractType{0}'.format(i)].to_numpy() if contractLimit[i] else None)

fleetAgeAvg = data['fleet_year'].to_numpy()
weightedAgeAvg = data['weighted_age'].to_numpy()

lesseeOneHot = {lesseeName: data[lesseeName].to_numpy() for lesseeName in data['customer'].value_counts().index}



nbv = np.nan_to_num(nbv)
cost = np.nan_to_num(cost)
fleetAgeAvg = np.nan_to_num(fleetAgeAvg)
weightedAgeAvg = np.nan_to_num(weightedAgeAvg)




print('Time for processing data', time.time() - mid_time)
mid_time = time.time()

print("==============================================================")
print('Model preparing...')
def SortTop(l, n):
    topN = heapq.nlargest(n, l, key=lambda x:x[1])
    return np.sum(np.stack([lesseeOneHot[topN[i][0]] for i in range(n)]), axis=0)

var = np.array([LpVariable('container_{0}'.format(i), lowBound=0, cat=LpBinary) for i in range(nbv.shape[0])])
prob = LpProblem("MyProblem", LpMaximize if maxOrMin else LpMinimize)

warmProb = LpProblem("WarmProblem", LpMaximize)
warmProb += lpSum(var * 1)
warmProb.solve(PULP_CBC_CMD(msg = False, timeLimit=1))

# objective function 
if NbvCost:
    prob += lpSum(var * nbv)
else:
    prob += lpSum(var * cost)
# constraints
numSelected = lpSum(var) # num of selected containers

# nbv
if maxTotalNbv:
    prob += lpSum(var * nbv) <= maxTotalNbv, "MaxNBV"
    print(1)
if minTotalNbv:
    prob += lpSum(var * nbv) >= minTotalNbv, "MinNBV"
    print(2)
    
# cost
if maxTotalCost:
    prob += lpSum(var * cost) <= maxTotalCost, "MaxCost"
    print(3)
if minTotalCost:
    prob += lpSum(var * cost) >= minTotalCost, "MinCost"
    print(4)

# status
if OnHireLimit:
    prob += lpSum(var * onHireStatus) >= OnHireLimit * numSelected, "OnHire"
    print(5)
if OffHireLimit:
    prob += lpSum(var * offHireStatus) <= OffHireLimit * numSelected, "OffHire"
    print(6)
if NoneHireLimit:
    prob += lpSum(var * noneHireStatus) <= NoneHireLimit * numSelected, "NoneHire"
    print(7) 
  
# container age
if fleetAgeAvgLimit:
    print(8)
    if fleetAgeAvgGeq:
        prob += lpSum(var * fleetAgeAvg) >= fleetAgeAvgLimit * numSelected, "FleetAgeAvg>"
    else:
        prob += lpSum(var * fleetAgeAvg) <= fleetAgeAvgLimit * numSelected, "FleetAgeAvg<"
for i in range(numLimit):
    if fleetAgeLimit[i]:
        print(9, i)
        if fleetAgeGeq[i]:
            prob += lpSum(var * fleetAge[i]) >= fleetAgeLimit[i] * numSelected, "FleetAge{0}>".format(i)
        else:
            prob += lpSum(var * fleetAge[i]) <= fleetAgeLimit[i] * numSelected, "FleetAge{0}<".format(i)

# weighted age
if weightedAgeAvgLimit:
    print(10)
    if weightedAgeAvgGeq:
        prob += lpSum(var * weightedAgeAvg) >= weightedAgeAvgLimit * numSelected, "WeightedAgeAvg>"
    else:
        prob += lpSum(var * weightedAgeAvg) <= weightedAgeAvgLimit * numSelected, "WeightedAgeAvg<"
for i in range(numLimit):
    if weightedAgeLimit[i]:
        print(11, i)
        if weightedAgeGeq[i]:
            prob += lpSum(var * weightedAge[i]) >= weightedAgeLimit[i] * numSelected, "WeightedAge{0}>".format(i)
        else:
            prob += lpSum(var * weightedAge[i]) <= weightedAgeLimit[i] * numSelected, "WeightedAge{0}<".format(i)

# product
for i in range(numLimit):
    if productLimit[i]:
        print(12, i)
        if productGeq[i]:
            prob += lpSum(var * product[i]) >= productLimit[i] * numSelected, "Product{0}>".format(i)
        else:
            prob += lpSum(var * product[i]) <= productLimit[i] * numSelected, "Product{0}<".format(i)

# lessee
for i in range(numLimit):
    if lesseeLimit[i] and lesseeType[i] in lesseeOneHot:
        print(13, i)
        if lesseeGeq[i]:
            prob += lpSum(var * lesseeOneHot[lesseeType[i]]) >= lesseeLimit[i] * numSelected, "Lessee{0}>".format(i)
        else:
            prob += lpSum(var * lesseeOneHot[lesseeType[i]]) <= lesseeLimit[i] * numSelected, "Lessee{0}<".format(i)


# find top1, top2, top3
for i in range(min(3, len(lesseeOneHot))):
    if topLesseeLimit[i]:
        print(14, i)
        if topLesseeGeq[i]:
            prob += lpSum(var * SortTop(list({j: value(lpSum(var * lesseeOneHot[j])) for j in lesseeOneHot.keys()}.items()), i+1)) >= topLesseeLimit[i] * numSelected, "Top{0}>".format(i+1)
        else:
            prob += lpSum(var * SortTop(list({j: value(lpSum(var * lesseeOneHot[j])) for j in lesseeOneHot.keys()}.items()), i+1)) <= topLesseeLimit[i] * numSelected, "Top{0}<".format(i+1)
            # prob += sum(heapq.nlargest([value(lpSum(var * lesseeOneHot[j])) for j in lesseeOneHot.keys()], i+1)) <= topLesseeLimit[i] * numSelected, "Top{0}<".format(i+1)

# contract type
for i in range(numLimit):
    if contractLimit[i]:
        print(15, i)
        if contractGeq[i]:
            prob += lpSum(var * contract[i]) >= contractLimit[i] * numSelected, "ContractType{0}>".format(i)
        else:
            prob += lpSum(var * contract[i]) <= contractLimit[i] * numSelected, "ContractType{0}<".format(i)


print('Time for model preparing', time.time() - mid_time)
print("==============================================================")
print('Model running...')
solver = PULP_CBC_CMD(msg = True, timeLimit=TIMEOUT)
try:
    prob.solve(solver)
except PulpSolverError:
    print('Nan value is not allowed in model. Need data cleaning.')
except Exception as e:
    print(e) 


print("==============================================================")
# print(prob)
print("status:", LpStatus[prob.status])
print("==============================================================")
print("target value: ", value(prob.objective))

# if solution is found
if prob.status == 1 or prob.status == 2:    
    result = np.array([var[i].varValue for i in range(len(var))])
    print(int(sum(result)), '/', len(result), 'containers are selected.')
    print('======================================================================')
    print("nbv: {0} between {1} - {2}".format(round(sum(result * nbv), 2), minTotalNbv, maxTotalNbv))
    print("cost: {0} between {1} - {2}".format(round(sum(result * cost), 2), minTotalCost, maxTotalCost))
    print('billing status:')
    if OnHireLimit:
        print('\t onhire {0}, -- {1}'.format(round(sum(result * onHireStatus)/sum(result), 2), OnHireLimit))
    if OffHireLimit:
        print('\t offhire {0}, -- {1}'.format(round(sum(result * offHireStatus)/sum(result), 2), OffHireLimit))
    if NoneHireLimit:
        print('\t none {0}, -- {1}'.format(round(sum(result * noneHireStatus)/sum(result), 2), NoneHireLimit))

    print("container age:")
    print('\t container average age is {0}, -- {1}'.format(round(sum(result * fleetAgeAvg)/sum(result), 2), fleetAgeAvgLimit))
    for i in range(numLimit):
        if fleetAgeLimit[i]:
            print("\t container age from {0} to {1} is {2}, -- {3}:".format(fleetAgeLowBound[i], fleetAgeUpBound[i], round(sum(result * fleetAge[i])/sum(result), 2), fleetAgeLimit[i]))

    print("weighted age:")
    print('\t weighted average age is {0}, -- {1}'.format(round(sum(result * weightedAgeAvg)/sum(result), 2), weightedAgeAvgLimit))
    for i in range(numLimit):
        if weightedAgeLimit[i]:
            print("\t weighted age from {0} to {1} is {2}, -- {3}:".format(weightedAgeLowBound[i], weightedAgeUpBound[i], round(sum(result * weightedAge[i])/sum(result), 2), weightedAgeLimit[i]))    

    print("product:")
    for i in range(numLimit):
        if productLimit[i]:
            print("\t product {0} is {1}, -- {2}:".format(productType[i], round(sum(result * product[i])/sum(result), 2), productLimit[i]))    
    
    print("lessee:")
    for i in range(numLimit):
        if lesseeLimit[i]:
            print("\t lessee {0} is {1}, -- {2}:".format(lesseeType[i], round(sum(result * lesseeOneHot[lesseeType[i]])/sum(result), 2), lesseeLimit[i]))    

    print('Top lessee:')
    numLessee = {lesseeName: value(lpSum(var * lesseeOneHot[lesseeName])) for lesseeName in data['customer'].value_counts().index}
    sortedLessee = list(numLessee.items())
    top3Lessee = heapq.nlargest(3, sortedLessee, key=lambda x:x[1])
    if topLesseeLimit[0]:
        print('\t top 1 {0} is {1}, -- {2}'.format(top3Lessee[0][0], round(top3Lessee[0][1] / sum(result), 2), topLesseeLimit[0]))
    if topLesseeLimit[1]:
        print('\t top 2 {0} {1} is {2}, -- {3}'.format(top3Lessee[0][0], top3Lessee[1][0], round((top3Lessee[0][1] + top3Lessee[1][1]) / sum(result), 2), topLesseeLimit[1]))
    if topLesseeLimit[2]:
        print('\t top 3 {0} {1} {2} is {3}, -- {4}'.format(top3Lessee[0][0], top3Lessee[1][0], top3Lessee[2][0], round((top3Lessee[0][1] + top3Lessee[1][1] + top3Lessee[2][1])/ sum(result), 2), topLesseeLimit[2]))

    print("contract type:")
    for i in range(numLimit):
        if contractLimit[i]:
            print("\t contract type {0} is {1}, -- {2}:".format(contractType[i], round(sum(result * contract[i])/sum(result), 2), contractLimit[i])) 



if prob.status == 1 or prob.status == 2:
    container_list = []
    for i in range(len(result)):
        if result[i]:
            container_list.append(data['unit_id_fz'][i])

    sqlOutput = """
            INSERT INTO fll_t_dw.biz_fir_asset_package (unit_id, query_id, id)
            VALUES %s
            """.format(queryID)
    try:
        conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
        conn.autocommit = True
        cur = conn.cursor()
        print('Writing data...')
        
        values_list = []
        for i in range(n):
            if result[i]:
                values_list.append((data['unit_id_fz'][i], queryID, uuid.uuid1().hex))

        psycopg2.extras.execute_values(cur, sqlOutput, values_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        print("Writing data to GreenPlum Failed!")


print('Total time cost', time.time() - start_time)
