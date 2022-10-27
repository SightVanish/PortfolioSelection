import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize, LpBinary, LpStatus, value, PulpSolverError
from pulp.apis import PULP_CBC_CMD
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
import collections
total_time = time.time()
numLimit = 5 # maximum num of constraints in each condition

if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)

def ReportStatus(msg, flag):
    sql = "update fll_t_dw.biz_fir_query_parameter_definition set python_info_data='{0}', success_flag='{1}' where id='{2}'".format(msg, flag, queryID)
    print("============================================================================================================================")
    print("Reporting issue:", msg)
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    conn.close()

try:
    print("============================================================================================================================")
    print('Parameters reading...')
    sqlParameter = "select python_json from fll_t_dw.biz_fir_query_parameter_definition where success_flag='T'"
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
    param = pd.read_sql(sqlParameter, conn)
    paramDict = json.loads(param['python_json'][0])
except Exception as e:
    print("Loading Parameters from GreenPlum Failed!\n", e)
    exit(1)

try:
    print('Data loading...')
    queryID = paramDict['query_id']
    print('Query ID:', queryID)
    sqlInput = "select billing_status_fz as billing, unit_id_fz, product, fleet_year_fz as fleet_year, contract_cust_id as customer, contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age, query_id from fll_t_dw.biz_ads_fir_pkg_data WHERE query_id='{0}'".format(queryID) 
    data = pd.read_sql(sqlInput, conn)
    conn.close()
except Exception as e:
    print(e)
    ReportStatus("Loading Data from GreenPlum Failed!", 'F')
    exit(1)

if data.shape[0] == 0:
    ReportStatus("No Available Data!", 'F')
    exit(1)


print("==============================================================")
try:
    queryID = paramDict['query_id']
    initialQuery = paramDict['initial_query']
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
    statusType = [None for _ in range(numLimit)]
    statusLimit = [None for _ in range(numLimit)]
    statusGeq = [None for _ in range(numLimit)]

    minTotalNbv = param['totalNBVFrom'] if param['totalNBVFrom'] else None
    maxTotalNbv = param['totalNBVTo'] if param['totalNBVTo'] else None

    minTotalCost = param['totalCostFrom'] if param['totalCostFrom'] else None
    maxTotalCost = param['totalCostTo'] if param['totalCostTo'] else None

    fleetAgeAvgLimit = param['containersAge']['average']['averageContainersAge'] if param['containersAge']['average']['averageContainersAge'] else None
    fleetAgeAvgGeq = param['containersAge']['average']['symbol']
    for i in range(len(param['containersAge']['list'])):
        fleetAgeLowBound[i] = param['containersAge']['list'][i]['containersAgeFrom']
        fleetAgeUpBound[i] = param['containersAge']['list'][i]['containersAgeTo']
        fleetAgeLimit[i] = param['containersAge']['list'][i]['percent'] / 100
        fleetAgeGeq[i] = param['containersAge']['list'][i]['symbol']

    weightedAgeAvgLimit = param['weightedAge']['average']['averageWeighedAge'] if param['weightedAge']['average']['averageWeighedAge'] else None
    weightedAgeAvgGeq = param['weightedAge']['average']['symbol']
    for i in range(len(param['weightedAge']['list'])):
        weightedAgeLowBound[i] = param['weightedAge']['list'][i]['weightedAgeFrom']
        weightedAgeUpBound[i] = param['weightedAge']['list'][i]['weightedAgeTo']
        weightedAgeLimit[i] = param['weightedAge']['list'][i]['percent'] / 100
        weightedAgeGeq[i] = param['weightedAge']['list'][i]['symbol']

    topLesseeLimit = [param['TopLessee'][0]['percent'] / 100 if param['TopLessee'][0]['Top1'] else None,
                param['TopLessee'][1]['percent'] / 100 if param['TopLessee'][1]['Top2'] else None,
                param['TopLessee'][2]['percent'] / 100 if param['TopLessee'][2]['Top3'] else None]
    topLesseeGeq = [param['TopLessee'][0]['symbol'] if param['TopLessee'][0]['Top1'] else None,
                param['TopLessee'][1]['symbol'] if param['TopLessee'][1]['Top2'] else None,
                param['TopLessee'][2]['symbol'] if param['TopLessee'][2]['Top3'] else None]
    for i in range(len(param['lessee'])):
        lesseeType[i] = param['lessee'][i]['lessee']
        lesseeLimit[i] = param['lessee'][i]['percent'] / 100
        lesseeGeq[i] = param['lessee'][i]['symbol']

    for i in range(len(param['status'])):
        statusType[i] = param['status'][i]['statusType']
        statusLimit[i] = param['status'][i]['percent'] / 100
        statusGeq[i] = param['status'][i]['symbol']

    for i in range(len(param['product'])):
        productType[i] = param['product'][i]['productType']
        productLimit[i] = param['product'][i]['percent'] / 100
        productGeq[i] = param['product'][i]['symbol']

    for i in range(len(param['contractType'])):
        contractType[i] = param['contractType'][i]['contractType']
        contractLimit[i] = param['contractType'][i]['percent'] / 100
        contractGeq[i] = param['contractType'][i]['symbol']
except Exception as e:
    print(e)
    ReportStatus('Parsing Paramters Failed!', 'F')
    exit(1)

print("==============================================================")
print('Data processing...')
# Billing Status
data['OnHireStatus'] = data['billing'].apply(lambda x: 1 if x=='ON' else 0)
data['OffHireStatus'] = data['billing'].apply(lambda x: 1 if x=='OF' else 0)
data['NoneStatus'] = data['billing'].apply(lambda x: 1 if (x!='ON' and x!='OF') else 0)

# ONE HOT -- all lessee
for lesseeName in data['customer'].value_counts().index:
    data[lesseeName] = data['customer'].apply(lambda x: 1 if x==lesseeName else 0)

for i in range(numLimit):
    # Container Age
    if fleetAgeLimit[i]:
        column_name = 'FleetAge{0}'.format(i)
        data[column_name] = data['fleet_year'].apply(lambda x: 1 if fleetAgeLowBound[i]<=x<fleetAgeUpBound[i] else 0)
    # Weighted Age
    if weightedAgeLimit[i]:
        column_name = 'WeightedAge{0}'.format(i)
        data[column_name] = data['weighted_age'].apply(lambda x: 1 if weightedAgeLowBound[i]<=x<weightedAgeUpBound[i] else 0)
    # Product Type
    if productLimit[i]:
        column_name = 'ProductType{0}'.format(i)
        data[column_name] = data['product'].apply(lambda x: 1 if x in productType[i] else 0)
    # Contract Type
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
ceu = data['ceu'].to_numpy()

def BuildModel(topLesseeCandidate, TopConstraints):
    start_time = time.time()
    print("==============================================================")
    print('Model preparing...')
    var = np.array([LpVariable('container_{0}'.format(i), lowBound=0, cat=LpBinary) for i in range(nbv.shape[0])])
    prob = LpProblem("MyProblem", LpMaximize if maxOrMin else LpMinimize)

    # objective function 
    if NbvCost:
        prob += lpSum(var * nbv)
    else:
        prob += lpSum(var * cost)

    # constraints
    numSelected = lpSum(var) # num of selected containers
    ceuSelected = lpSum(var * ceu)
    # nbv
    if maxTotalNbv:
        prob += lpSum(var * nbv) <= maxTotalNbv, "MaxNBV"
        print('Set Max Nbv')
    if minTotalNbv:
        prob += lpSum(var * nbv) >= minTotalNbv, "MinNBV"
        print('Set Min Nbv')
    # cost
    if maxTotalCost:
        prob += lpSum(var * cost) <= maxTotalCost, "MaxCost"
        print('Set Max Cost')
    if minTotalCost:
        prob += lpSum(var * cost) >= minTotalCost, "MinCost"
        print('Set min Cost')
    # status
    for i in range(numLimit):
        if statusType[i]:
            print('Set Status Limit', i)
            if statusType[i] == 'ON':
                if statusGeq[i]:
                    prob += lpSum(var * onHireStatus * ceu) >= statusLimit[i] * ceuSelected, "OnHire{0}>".format(i)
                else:
                    prob += lpSum(var * onHireStatus * ceu) <= statusLimit[i] * ceuSelected, "OnHire{0}<".format(i)
            if statusType[i] == 'OF':
                if statusGeq[i]:
                    prob += lpSum(var * offHireStatus * ceu) >= statusLimit[i] * ceuSelected, "OffHire{0}>".format(i)
                else:
                    prob += lpSum(var * offHireStatus * ceu) <= statusLimit[i] * ceuSelected, "OffHire{0}<".format(i)
            if statusType[i] == 'None':
                if statusGeq[i]:
                    prob += lpSum(var * noneHireStatus * ceu) >= statusLimit[i] * ceuSelected, "NoneHire{0}>".format(i)
                else:
                    prob += lpSum(var * noneHireStatus * ceu) <= statusLimit[i] * ceuSelected, "NoneHire{0}<".format(i)
    # container age
    if fleetAgeAvgLimit:
        print('Set Container Average Age Limit')
        if fleetAgeAvgGeq:
            prob += lpSum(var * fleetAgeAvg) >= fleetAgeAvgLimit * numSelected, "FleetAgeAvg>"
        else:
            prob += lpSum(var * fleetAgeAvg) <= fleetAgeAvgLimit * numSelected, "FleetAgeAvg<"
    for i in range(numLimit):
        if fleetAgeLimit[i]:
            print('Set Container Age Limit', i)
            if fleetAgeGeq[i]:
                prob += lpSum(var * fleetAge[i]) >= fleetAgeLimit[i] * numSelected, "FleetAge{0}>".format(i)
            else:
                prob += lpSum(var * fleetAge[i]) <= fleetAgeLimit[i] * numSelected, "FleetAge{0}<".format(i)
    # weighted age
    if weightedAgeAvgLimit:
        print('Set Weighted Average Age Limit')
        if weightedAgeAvgGeq:
            prob += lpSum(var * weightedAgeAvg) >= weightedAgeAvgLimit * ceuSelected, "WeightedAgeAvg>"
        else:
            prob += lpSum(var * weightedAgeAvg) <= weightedAgeAvgLimit * ceuSelected, "WeightedAgeAvg<"
    for i in range(numLimit):
        if weightedAgeLimit[i]:
            print('Set Weighted Age Limit', i)
            if weightedAgeGeq[i]:
                prob += lpSum(var * weightedAge[i]) >= weightedAgeLimit[i] * ceuSelected, "WeightedAge{0}>".format(i)
            else:
                prob += lpSum(var * weightedAge[i]) <= weightedAgeLimit[i] * ceuSelected, "WeightedAge{0}<".format(i)
    # product
    for i in range(numLimit):
        if productLimit[i]:
            print('Set Produdct Limit', i)
            if productGeq[i]:
                prob += lpSum(var * product[i] * ceu) >= productLimit[i] * ceuSelected, "Product{0}>".format(i)
            else:
                prob += lpSum(var * product[i] * ceu) <= productLimit[i] * ceuSelected, "Product{0}<".format(i)
    # lessee
    for i in range(numLimit):
        if lesseeLimit[i] and lesseeType[i] in lesseeOneHot:
            print('Set Lessee Limit', i)
            if lesseeGeq[i]:
                prob += lpSum(var * lesseeOneHot[lesseeType[i]] * ceu) >= lesseeLimit[i] * ceuSelected, "Lessee{0}>".format(i)
            else:
                prob += lpSum(var * lesseeOneHot[lesseeType[i]] * ceu) <= lesseeLimit[i] * ceuSelected, "Lessee{0}<".format(i)
   
    # TOP1 lessee
    if topLesseeLimit[0] and len(topLesseeCandidate) >= 1:
        for i in topLesseeCandidate:
            if {i} not in TopConstraints:
                print('Set Top1 Limit', i)
                prob += lpSum(var * ceu * lesseeOneHot[i]) <= topLesseeLimit[0] * ceuSelected, "Top1:{0}<".format(i)
                TopConstraints.append({i})
    # TOP2 lessee
    if topLesseeLimit[1] and len(topLesseeCandidate) >= 2:
        for i in topLesseeCandidate:
            for j in topLesseeCandidate:
                if {i, j} not in TopConstraints and len({i, j}) == 2:
                    print('Set Top2 Limit', i, j)
                    prob += lpSum(var * ceu * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j]]), axis=0)) <= topLesseeLimit[1] * ceuSelected, "Top2:{0}&{1}<".format(i,j)
                    TopConstraints.append({i, j})
    # TOP3 lessee
    if topLesseeLimit[2] and len(topLesseeCandidate) >= 3:
        for i in topLesseeCandidate:
            for j in topLesseeCandidate:
                for k in topLesseeCandidate:
                    if {i, j, k} not in TopConstraints and len({i, j, k}) == 3:
                        print('Set Top3 Limit', i, j, k)
                        prob += lpSum(var * ceu * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j], lesseeOneHot[k]]), axis=0)) <= topLesseeLimit[2] * ceuSelected, "Top3:{0}&{1}&{2}<".format(i,j,k)
                        TopConstraints.append({i, j, k})

    # contract type
    for i in range(numLimit):
        if contractLimit[i]:
            print('Set Contract Type Limit', i)
            if contractGeq[i]:
                prob += lpSum(var * contract[i] * ceu) >= contractLimit[i] * ceuSelected, "ContractType{0}>".format(i)
            else:
                prob += lpSum(var * contract[i] * ceu) <= contractLimit[i] * ceuSelected, "ContractType{0}<".format(i)

    return prob, var

def SolveModel(prob, var, timeLimit):
    start_time = time.time()
    print("==============================================================")
    print('Model solving...')
    # solve model
    solver = PULP_CBC_CMD(msg = True, timeLimit=timeLimit, threads=8)
    prob.solve(solver)
    print("==============================================================")
    print("status:", LpStatus[prob.status])
    print("==============================================================")
    print('Time Cost', time.time() - start_time)

    return prob, var, prob.status

def UpdateModel(prob, var, topLesseeCandidate, TopConstraints):
    print("==============================================================")
    print('Model updating...')
    ceuSelected = lpSum(var * ceu) # num of selected containers
    # TOP1 lessee
    if topLesseeLimit[0] and len(topLesseeCandidate) >= 1:
        for i in topLesseeCandidate:
            if {i} not in TopConstraints:
                print('Update Top1 Limit', i)
                prob += lpSum(var * ceu * lesseeOneHot[i]) <= topLesseeLimit[0] * ceuSelected, "Top1:{0}<".format(i)
                TopConstraints.append({i})
    # TOP2 lessee
    if topLesseeLimit[1] and len(topLesseeCandidate) >= 2:
        for i in topLesseeCandidate:
            for j in topLesseeCandidate:
                if {i, j} not in TopConstraints and len({i, j}) == 2:
                        print('Update Top2 Limit', i, j)
                        prob += lpSum(var * ceu * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j]]), axis=0)) <= topLesseeLimit[1] * ceuSelected, "Top2:{0}&{1}<".format(i,j)
                        TopConstraints.append({i, j})
    # TOP3 lessee
    if topLesseeLimit[2] and len(topLesseeCandidate) >= 3:
        for i in topLesseeCandidate:
            for j in topLesseeCandidate:
                    for k in topLesseeCandidate:
                            if {i, j, k} not in TopConstraints and len({i, j, k}) == 3:
                                print('Update Top3 Limit', i, j, k)
                                prob += lpSum(var * ceu * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j], lesseeOneHot[k]]), axis=0)) <= topLesseeLimit[2] * ceuSelected, "Top3:{0}&{1}&{2}<".format(i,j,k)
                                TopConstraints.append({i, j, k})
    return prob, var


TopConstraints = []

topLesseeCandidate = set(data['customer'].value_counts().keys()[:3])
print('Top Lessee Candidates:', topLesseeCandidate)
prob, var = BuildModel(topLesseeCandidate, TopConstraints)

while True:
    prob, var, s = SolveModel(prob, var, 200 * len(topLesseeCandidate))
    if prob.status != 1:
        print('Algorithm Failed!')
        break
    result = np.array([var[i].varValue for i in range(len(var))])
    top3Lessee = heapq.nlargest(3, [(lesseeName, sum(result * lesseeOneHot[lesseeName])) for lesseeName in data['customer'].value_counts().index], key=lambda x:x[1])
    top3 = set(i[0] for i in top3Lessee)
    if topLesseeCandidate >= top3:
        print('Algorithm Succeeded!')
        break
    else:
        print("============================================================================================================================")
        print('Recurse...xD')
        topLesseeCandidate = topLesseeCandidate.union(top3)
        print('Top3 lessee:', top3)
        print('Top Lessee Candidates:', topLesseeCandidate)
        prob, var = UpdateModel(prob, var, topLesseeCandidate, TopConstraints)

if 1:    
    result = np.array([var[i].varValue for i in range(len(var))])
    print(set(result))
    # result = np.array([1 if var[i].varValue==1 else 0 for i in range(len(var))])
    # print(set(result))
    print(int(sum(result)), '/', len(result), 'containers are selected.')
    print('======================================================================')
    print("nbv: {0} between {1} - {2}".format(round(sum(result*nbv), 4), minTotalNbv, maxTotalNbv))
    print("cost: {0} between {1} - {2}".format(round(sum(result*cost), 4), minTotalCost, maxTotalCost))
    print('billing status:')
    for i in range(numLimit):
        if statusType[i]:
            if statusType[i] == 'ON':
                print('\t OnHire is {0}, -- {1}'.format(round(sum(result*onHireStatus*ceu)/sum(result*ceu), 4), statusLimit[i]))
            if statusType[i] == 'OF':
                print('\t OffHire is {0}, -- {1}'.format(round(sum(result*offHireStatus*ceu)/sum(result*ceu), 4), statusLimit[i]))
            if statusType[i] == 'None':
                print('\t NoneHire is {0}, -- {1}'.format(round(sum(result*noneHireStatus*ceu)/sum(result*ceu), 4), statusLimit[i]))

    print("container age:")
    if fleetAgeAvgLimit:
        print('\t container average age is {0}, -- {1}'.format(round(sum(result*fleetAgeAvg)/sum(result), 4), fleetAgeAvgLimit))
    for i in range(numLimit):
        if fleetAgeLimit[i]:
            print("\t container age from {0} to {1} is {2}, -- {3}:".format(fleetAgeLowBound[i], fleetAgeUpBound[i], round(sum(result*fleetAge[i])/sum(result), 4), fleetAgeLimit[i]))

    print("weighted age:")
    if weightedAgeAvgLimit:
        print('\t weighted average age is {0}, -- {1}'.format(round(sum(result * weightedAgeAvg)/sum(result*ceu), 4), weightedAgeAvgLimit))
    for i in range(numLimit):
        if weightedAgeLimit[i]:
            print("\t weighted age from {0} to {1} is {2}, -- {3}:".format(weightedAgeLowBound[i], weightedAgeUpBound[i], round(sum(result * weightedAge[i])/sum(result*ceu), 4), weightedAgeLimit[i]))    

    print("product:")
    for i in range(numLimit):
        if productLimit[i]:
            print("\t product {0} is {1}, -- {2}:".format(productType[i], round(sum(result*product[i]*ceu)/sum(result*ceu), 4), productLimit[i]))    
    
    print("lessee:")
    for i in range(numLimit):
        if lesseeLimit[i]:
            print("\t lessee {0} is {1}, -- {2}:".format(lesseeType[i], round(sum(result*lesseeOneHot[lesseeType[i]]*ceu)/sum(result*ceu), 4), lesseeLimit[i]))    

    print('Top lessee:')
    numLessee = {lesseeName: value(lpSum(var*lesseeOneHot[lesseeName]*ceu)) for lesseeName in data['customer'].value_counts().index}
    sortedLessee = list(numLessee.items())
    top3Lessee = heapq.nlargest(3, sortedLessee, key=lambda x:x[1])
    if topLesseeLimit[0]:
        print('\t top 1 {0} is {1}, -- {2}'.format(top3Lessee[0][0], top3Lessee[0][1]/sum(result*ceu), topLesseeLimit[0]))
    if topLesseeLimit[1]:
        if len(top3Lessee) >= 2:
            print('\t top 2 {0} {1} is {2}, -- {3}'.format(top3Lessee[0][0], top3Lessee[1][0], (top3Lessee[0][1]+top3Lessee[1][1])/sum(result*ceu), topLesseeLimit[1]))
        else:
            print('Only one lessee.')
    if topLesseeLimit[2]:
        if len(top3Lessee) >= 3:
            print('\t top 3 {0} {1} {2} is {3}, -- {4}'.format(top3Lessee[0][0], top3Lessee[1][0], top3Lessee[2][0], (top3Lessee[0][1]+top3Lessee[1][1]+top3Lessee[2][1])/sum(result*ceu), topLesseeLimit[2]))
        else:
            print('Only two lessee.')
            
    print("contract type:")
    for i in range(numLimit):
        if contractLimit[i]:
            print("\t contract type {0} is {1}, -- {2}:".format(contractType[i], round(sum(result*contract[i]*ceu)/sum(result*ceu), 4), contractLimit[i])) 


print("============================================================================================================================")
print('Total Time Cost:', time.time() - total_time)

print('Output to ./demo_result.csv')
outputData = data[['unit_id', 'contract_num', 'cost', 'product', 'customer', 'contract', 'nbv', 'billing', 'fleet_year', 'weighted_age', 'ceu']].copy()
outputData.columns = ['Unit Id Fz', 'Contract Num', 'Cost', 'Product', 'Contract Cust Id', 'Contract Lease Type', 'Nbv', 'Billing Status Fz', 'Fleet Year Fz', 'Age x CEU', 'Ceu Fz']
outputData.insert(loc=0, column="Selected", value=result)
outputData.to_csv('./demo_result.csv')