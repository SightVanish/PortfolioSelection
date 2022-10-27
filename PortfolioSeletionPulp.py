import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize, LpBinary, LpStatus, value, PulpSolverError
from pulp.apis import PULP_CBC_CMD
import time
import heapq
import json
import psycopg2
import psycopg2.extras
import warnings
import sys
import uuid
numLimit = 5 # maximum num of constraints in each condition
timeLimit = 150
if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)
def ReportStatus(msg, flag, queryID):
    sql = "update fll_t_dw.biz_fir_query_parameter_definition set python_info_data='{0}', success_flag='{1}' where id='{2}'".format(msg, flag, queryID)
    print("============================================================================================================================")
    print("Reporting issue:", msg)
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    conn.close()
def DecideBasis(basis, var, ceu, teu, nbv, cost, queryID):
    if basis == 'ceu':
        return ceu, lpSum(var * ceu)
    elif basis == 'teu':
        return ceu, lpSum(var * teu)
    elif basis == 'nbv':
        return ceu, lpSum(var * nbv)
    elif basis == 'cost':
        return ceu, lpSum(var * cost)
    else:
        ReportStatus('Basis is not valid!', 'F', queryID)
def DecideStatus(status, onHireStatus, offHireStatus, noneHireStatus, queryID):
    if status == 'ON':
        return "OnHire", onHireStatus
    elif status == 'OF':
        return "OffHire", offHireStatus
    elif status == 'None':
        return "NoneHire", noneHireStatus
    else:
        ReportStatus('Status is not valid!', 'F', queryID)
def ValidTopConstraints(topLesseeLimit, topLesseeCandidate, top3):
    # valid top3
    if topLesseeLimit[2]:
        if not (set(top3[:3]) <= topLesseeCandidate):
            return False, set.union(topLesseeCandidate, top3[:3])
    # valid top2
    if topLesseeLimit[1]:
        if not (set(top3[:3]) <= topLesseeCandidate):
            return False, set.union(topLesseeCandidate, top3[:3])
    # valid top1
    if topLesseeLimit[0]:
        if not (set(top3[:1]) <= topLesseeCandidate):
            return False, set.union(topLesseeCandidate, top3[:1])
    return True, topLesseeCandidate

# try:
#     print('Parameters reading...')
#     sqlParameter = "select python_json, id from fll_t_dw.biz_fir_query_parameter_definition where success_flag='T'"
#     conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
#     paramInput = pd.read_sql(sqlParameter, conn)
#     if paramInput.shape[0] == 0:
#         raise Exception('No Valid Query Request is Found!')
#     elif paramInput.shape[0] > 1:
#         raise Exception('More than One Valid Query Requests are Found!')
#     queryID = paramInput['id'][0]
#     param = json.loads(paramInput['python_json'][0])
# except Exception as e:
#     print("Loading Parameters from GreenPlum Failed!\n", e)
#     exit(1)

# try:
#     print('Data loading...')
#     print('Query ID:', queryID)
#     sqlInput = """
#         select billing_status_fz as billing, unit_id_fz, product, fleet_year_fz as fleet_year, contract_cust_id as customer, \
#         contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age, query_id, ceu_fz as ceu, teu_fz as teu
#         from fll_t_dw.biz_ads_fir_pkg_data WHERE query_id='{0}'
#     """.format(queryID) 
#     data = pd.read_sql(sqlInput, conn)

#     if data.shape[0] == 0:
#         raise Exception("No Data Available!")
#     conn.close()
# except Exception as e:
#     print(e)
#     ReportStatus("Loading Data from GreenPlum Failed!", 'F', queryID)
#     exit(1)

# TEST
queryID = 'testidddddddddddddddddddddddddddd'
with open("./parameterDemo1.json") as f:
    param = json.load(f)
print('Data Loading...')
rawData = pd.read_excel(io='./test_data_with_constraints.xlsb', sheet_name='数据', engine='pyxlsb')
data = rawData[['Unit Id Fz', 'Contract Num', 'Cost', 'Product', 'Contract Cust Id', 'Contract Lease Type', 'Nbv', 'Billing Status Fz', 'Fleet Year Fz', 'Age x CEU', 'Ceu Fz', 'Teu Fz']].copy()
data.columns = ['unit_id', 'contract_num', 'cost', 'product', 'customer', 'contract', 'nbv', 'billing', 'fleet_year', 'weighted_age', 'ceu', 'teu']


print("==============================================================")
print('Parameters parsing...')
try:
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

    minTotalNbv = param['totalNBVFrom']
    maxTotalNbv = param['totalNBVTo']

    minTotalCost = param['totalCostFrom']
    maxTotalCost = param['totalCostTo']

    topLesseeLimit = [
        param['lessee']['TopLessee']['list'][0]['percent'] / 100,
        param['lessee']['TopLessee']['list'][1]['percent'] / 100,
        param['lessee']['TopLessee']['list'][2]['percent'] / 100]
    topLesseeGeq = [
        param['lessee']['TopLessee']['list'][0]['symbol'],
        param['lessee']['TopLessee']['list'][1]['symbol'],
        param['lessee']['TopLessee']['list'][2]['symbol']]

    fleetAgeAvgLimit = param['containersAge']['average']['averageContainersAge']
    fleetAgeAvgGeq = param['containersAge']['average']['symbol']
    fleetAgeBasis = param['containersAge']['basis']
    for i in range(len(param['containersAge']['list'])):
        fleetAgeLowBound[i] = param['containersAge']['list'][i]['containersAgeFrom']
        fleetAgeUpBound[i] = param['containersAge']['list'][i]['containersAgeTo']
        fleetAgeLimit[i] = param['containersAge']['list'][i]['percent'] / 100
        fleetAgeGeq[i] = param['containersAge']['list'][i]['symbol']

    weightedAgeAvgLimit = param['weightedAge']['average']['averageWeighedAge']
    weightedAgeAvgGeq = param['weightedAge']['average']['symbol']
    weightedAgeBasis = param['weightedAge']['basis']
    for i in range(len(param['weightedAge']['list'])):
        weightedAgeLowBound[i] = param['weightedAge']['list'][i]['weightedAgeFrom']
        weightedAgeUpBound[i] = param['weightedAge']['list'][i]['weightedAgeTo']
        weightedAgeLimit[i] = param['weightedAge']['list'][i]['percent'] / 100
        weightedAgeGeq[i] = param['weightedAge']['list'][i]['symbol']

    lesseeBasis = param['lessee']['basis']
    for i in range(len(param['lessee']['list'])):
        lesseeType[i] = param['lessee']['list'][i]['lessee']
        lesseeLimit[i] = param['lessee']['list'][i]['percent'] / 100
        lesseeGeq[i] = param['lessee']['list'][i]['symbol']

    statusBasis = param['status']['basis']
    for i in range(len(param['status']['list'])):
        statusType[i] = param['status']['list'][i]['statusType']
        statusLimit[i] = param['status']['list'][i]['percent'] / 100
        statusGeq[i] = param['status']['list'][i]['symbol']

    productBasis = param['product']['basis']
    for i in range(len(param['product']['list'])):
        productType[i] = param['product']['list'][i]['productType']
        productLimit[i] = param['product']['list'][i]['percent'] / 100
        productGeq[i] = param['product']['list'][i]['symbol']

    contractBasis = param['contractType']['basis']
    for i in range(len(param['contractType']['list'])):
        contractType[i] = param['contractType']['list'][i]['contractType']
        contractLimit[i] = param['contractType']['list'][i]['percent'] / 100
        contractGeq[i] = param['contractType']['list'][i]['symbol']
except Exception as e:
    print(e)
    ReportStatus('Parsing Paramters Failed!', 'F', queryID)
    exit(1)

print("==============================================================")
print('Data processing...')
try:
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
            data[column_name] = data['fleet_year'].apply(lambda x: 1 if fleetAgeLowBound[i]<=x<=fleetAgeUpBound[i] else 0)
        # Weighted Age
        if weightedAgeLimit[i]:
            column_name = 'WeightedAge{0}'.format(i)
            data[column_name] = data['weighted_age'].apply(lambda x: 1 if weightedAgeLowBound[i]<=x<=weightedAgeUpBound[i] else 0)
        # Product Type
        if productLimit[i]:
            column_name = 'ProductType{0}'.format(i)
            data[column_name] = data['product'].apply(lambda x: 1 if x in productType[i] else 0)
        # Contract Type
        if contractLimit[i]:
            column_name = 'ContractType{0}'.format(i)
            data[column_name] = data['contract'].apply(lambda x: 1 if x in contractType[i] else 0)

    # convert data to numpy
    nbv = data['nbv'].to_numpy()
    cost = data['cost'].to_numpy()
    ceu = data['ceu'].to_numpy()
    teu = data['teu'].to_numpy()
    fleetAgeAvg = data['fleet_year'].to_numpy()
    weightedAgeAvg = data['weighted_age'].to_numpy()
    onHireStatus = data['OnHireStatus'].to_numpy()
    offHireStatus = data['OffHireStatus'].to_numpy()
    noneHireStatus = data['NoneStatus'].to_numpy()
    lesseeOneHot = {lesseeName: data[lesseeName].to_numpy() for lesseeName in data['customer'].value_counts().index}
    fleetAge = []
    weightedAge = []
    product = []
    contract = []
    for i in range(numLimit):
        fleetAge.append(data['FleetAge{0}'.format(i)].to_numpy() if fleetAgeLimit[i] else None)
        weightedAge.append(data['WeightedAge{0}'.format(i)].to_numpy() if weightedAgeLimit[i] else None)
        product.append(data['ProductType{0}'.format(i)].to_numpy() if productLimit[i] else None)
        contract.append(data['ContractType{0}'.format(i)].to_numpy() if contractLimit[i] else None)

except Exception as e:
    print(e)
    ReportStatus('Processing Data Failed!', 'F', queryID)
    exit(1)

def BuildModel(topLesseeCandidate, TopConstraints):
    start_time = time.time()
    print("==============================================================")
    print('Model preparing...')
    var = np.array([LpVariable('container_{0}'.format(i), lowBound=0, cat=LpBinary) for i in range(nbv.shape[0])])
    prob = LpProblem("MyProblem", LpMaximize if maxOrMin else LpMinimize)
    
    numSelected = lpSum(var)
    ceuSelected = lpSum(var * ceu)
    teuSelected = lpSum(var * teu)
    nbvSelected = lpSum(var * nbv)
    costSelected = lpSum(var * cost)

    # objective function 
    if NbvCost:
        prob += lpSum(var * nbv)
    else:
        prob += lpSum(var * cost)

    # constraints
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
        print('Set Min Cost')
    # container age
    if fleetAgeAvgLimit:
        print('Set Container Average Age Limit')
        if fleetAgeAvgGeq:
            prob += lpSum(var * fleetAgeAvg) >= fleetAgeAvgLimit * numSelected, "FleetAgeAvg>"
        else:
            prob += lpSum(var * fleetAgeAvg) <= fleetAgeAvgLimit * numSelected, "FleetAgeAvg<"
    if fleetAgeBasis:
        basis, basisSelected = DecideBasis(fleetAgeBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if fleetAgeLimit[i]:
                print('Set Container Age Limit', i)
                if fleetAgeGeq[i]:
                    prob += lpSum(var * fleetAge[i] * basis) >= fleetAgeLimit[i] * basisSelected, "FleetAge{0}>".format(i)
                else:
                    prob += lpSum(var * fleetAge[i] * basis) <= fleetAgeLimit[i] * basisSelected, "FleetAge{0}<".format(i)
    # weighted age
    if weightedAgeAvgLimit:
        print('Set Weighted Average Age Limit')
        if weightedAgeAvgGeq:
            prob += lpSum(var * weightedAgeAvg) >= weightedAgeAvgLimit * ceuSelected, "WeightedAgeAvg>"
        else:
            prob += lpSum(var * weightedAgeAvg) <= weightedAgeAvgLimit * ceuSelected, "WeightedAgeAvg<"
    if weightedAgeBasis:
        basis, basisSelected = DecideBasis(weightedAgeBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if weightedAgeLimit[i]:
                print('Set Weighted Age Limit', i)
                if weightedAgeGeq[i]:
                    prob += lpSum(var * weightedAge[i] * basis) >= \
                                weightedAgeLimit[i] * basisSelected, "WeightedAge{0}>".format(i)
                else:
                    prob += lpSum(var * weightedAge[i] * basis) <= \
                                weightedAgeLimit[i] * basisSelected, "WeightedAge{0}<".format(i)
    # lessee
    if lesseeBasis:
        basis, basisSelected = DecideBasis(lesseeBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if lesseeLimit[i] and lesseeType[i] in lesseeOneHot:
                print('Set Lessee Limit', i)
                if lesseeGeq[i]:
                    prob += lpSum(var * lesseeOneHot[lesseeType[i]] * basis) >= \
                                lesseeLimit[i] * basisSelected, "Lessee{0}>".format(i)
                else:
                    prob += lpSum(var * lesseeOneHot[lesseeType[i]] * basis) <= \
                                lesseeLimit[i] * basisSelected, "Lessee{0}<".format(i)
    # status
    if statusBasis:
        basis, basisSelected = DecideBasis(statusBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if statusType[i]:
                print('Set Status Limit', i)
                statusName, status = DecideStatus(statusType[i], onHireStatus, offHireStatus, noneHireStatus, queryID)
                if statusGeq[i]:
                    prob += lpSum(var * status * basis) >= statusLimit[i] * basisSelected, "{0}{1}>".format(statusName, i)
                else:
                    prob += lpSum(var * status * basis) <= statusLimit[i] * basisSelected, "{0}{1}<".format(statusName, i)
    # product
    if productBasis:
        basis, basisSelected = DecideBasis(productBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if productLimit[i]:
                print('Set Produdct Limit', i)
                if productGeq[i]:
                    prob += lpSum(var * product[i] * basis) >= productLimit[i] * basisSelected, "Product{0}>".format(i)
                else:
                    prob += lpSum(var * product[i] * basis) <= productLimit[i] * basisSelected, "Product{0}<".format(i)
    # contract type
    if contractBasis:
        basis, basisSelected = DecideBasis(contractBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if contractLimit[i]:
                print('Set Contract Type Limit', i)
                if contractGeq[i]:
                    prob += lpSum(var * contract[i] * basis) >= contractLimit[i] * basisSelected, "ContractType{0}>".format(i)
                else:
                    prob += lpSum(var * contract[i] * basis) <= contractLimit[i] * basisSelected, "ContractType{0}<".format(i)
    return prob, var

def SolveModel(prob, var, timeLimit):
    start_time = time.time()
    print("==============================================================")
    print('Model solving...')
    # solve model
    solver = PULP_CBC_CMD(msg=False, timeLimit=timeLimit, threads=8)
    prob.solve(solver)
    print("==============================================================")
    print("status:", LpStatus[prob.status])
    print("==============================================================")
    print('Time Cost', time.time() - start_time)

    return prob, var

def UpdateModel(prob, var, topLesseeCandidate, TopConstraints):
    print("==============================================================")
    print('Model updating...')
    # lessee
    if lesseeBasis:
        basis, basisSelected = DecideBasis(lesseeBasis, var, ceu, teu, nbv, cost, queryID)
      # TOP1 lessee
        if topLesseeLimit[0] and len(topLesseeCandidate) >= 1:
            for i in topLesseeCandidate:
                if {i} not in TopConstraints:
                    print('Set Top1 Limit', i)
                    TopConstraints.append({i})
                    if topLesseeGeq[0]:
                        prob += lpSum(var * lesseeOneHot[i] * basis) >= topLesseeLimit[0] * basisSelected, "Top1:{0}>".format(i)
                    else:
                        prob += lpSum(var * lesseeOneHot[i] * basis) <= topLesseeLimit[0] * basisSelected, "Top1:{0}<".format(i)
        # TOP2 lessee
        if topLesseeLimit[1] and len(topLesseeCandidate) >= 2:
            for i in topLesseeCandidate:
                for j in topLesseeCandidate:
                    if {i, j} not in TopConstraints and len({i, j}) == 2:
                        print('Set Top2 Limit', i, j)
                        TopConstraints.append({i, j})
                        if topLesseeGeq[1]:
                            prob += lpSum(var * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j]]), axis=0) * basis) >= \
                                        topLesseeLimit[1] * basisSelected, "Top2:{0}&{1}>".format(i,j)
                        else:
                            prob += lpSum(var * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j]]), axis=0) * basis) <= \
                                        topLesseeLimit[1] * basisSelected, "Top2:{0}&{1}<".format(i,j)
        # TOP3 lessee
        if topLesseeLimit[2] and len(topLesseeCandidate) >= 3:
            for i in topLesseeCandidate:
                for j in topLesseeCandidate:
                    for k in topLesseeCandidate:
                        if {i, j, k} not in TopConstraints and len({i, j, k}) == 3:
                            print('Set Top3 Limit', i, j, k)
                            TopConstraints.append({i, j, k})
                            if topLesseeGeq[2]:
                                prob += lpSum(var * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j], lesseeOneHot[k]]), axis=0) * basis) >= \
                                        topLesseeLimit[2] * basisSelected, "Top3:{0}&{1}&{2}>".format(i,j,k)
                            else:
                                prob += lpSum(var * np.sum(np.stack([lesseeOneHot[i], lesseeOneHot[j], lesseeOneHot[k]]), axis=0) * basis) <= \
                                        topLesseeLimit[2] * basisSelected, "Top3:{0}&{1}&{2}<".format(i,j,k)
    return prob, var

TopConstraints = []
topLesseeCandidate = set(data['customer'].value_counts().keys()[:3])
print('Top Lessee Candidates:', topLesseeCandidate)
prob, var = BuildModel(topLesseeCandidate, TopConstraints)
prob, var = UpdateModel(prob, var, topLesseeCandidate, TopConstraints)

while True:
    try:
        prob, var = SolveModel(prob, var, timeLimit * len(topLesseeCandidate)) # increase running time
    except PulpSolverError:
        print()
        ReportStatus('Nan Data IS Not Allowed in Model. Need Data Cleaning!', 'F', queryID)
        exit(1)
    # TODO: output part
    if prob.status != 1:
        print('Algorithm Failed!')
        break
    elif prob.status == 1:
        result = np.array([var[i].varValue for i in range(len(var))]) # get result value
        top3Lessee = heapq.nlargest(3, [(lesseeName, sum(result * lesseeOneHot[lesseeName])) for lesseeName in data['customer'].value_counts().index], key=lambda x:x[1])
        valid, topLesseeCandidate = ValidTopConstraints(topLesseeLimit, topLesseeCandidate, [l[0] for l in top3Lessee])
        print('Top3 lessee:', top3Lessee)
        print('Top Lessee Candidates:', topLesseeCandidate)
        if valid:
            print('Algorithm Succeeded! LOLLLLLLLLLLLLLL')
            break
        else:
            print("============================================================================================================================")
            print('Recurse...xD')
            prob, var = UpdateModel(prob, var, topLesseeCandidate, TopConstraints)


print("============================================================================================================================")
if 1:    
    result = np.array([var[i].varValue for i in range(len(var))])
    print('Result is Valid:', set(result) == 2)
    result = np.array([1 if var[i].varValue==1 else 0 for i in range(len(var))])
    print(int(sum(result)), '/', len(result), 'containers are selected.')
    print('======================================================================')
    print("nbv: {0} between {1} - {2}".format(round(sum(result*nbv), 4), minTotalNbv, maxTotalNbv))
    print("cost: {0} between {1} - {2}".format(round(sum(result*cost), 4), minTotalCost, maxTotalCost))
    
    print('billing status:', statusBasis)
    if statusBasis:
        basis, _ = DecideBasis(statusBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if statusType[i]:
                if statusType[i] == 'ON':
                    print('\t OnHire on is {0}, -- {1}'.format(round(sum(result*onHireStatus*basis)/sum(result*basis), 4), statusLimit[i]))
                if statusType[i] == 'OF':
                    print('\t OffHire is {0}, -- {1}'.format(round(sum(result*offHireStatus*basis)/sum(result*basis), 4), statusLimit[i]))
                if statusType[i] == 'None':
                    print('\t NoneHire is {0}, -- {1}'.format(round(sum(result*noneHireStatus*basis)/sum(result*basis), 4), statusLimit[i]))

    print("container age:", fleetAgeBasis)
    if fleetAgeAvgLimit:
        print('\t container average age is {0}, -- {1}'.format(round(sum(result*fleetAgeAvg)/sum(result), 4), fleetAgeAvgLimit))
    if fleetAgeBasis:
        basis, _ = DecideBasis(fleetAgeBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if fleetAgeLimit[i]:
                print("\t container age from {0} to {1} is {2}, -- {3}:".format(fleetAgeLowBound[i], fleetAgeUpBound[i], round(sum(result*fleetAge[i]*basis)/sum(result*basis), 4), fleetAgeLimit[i]))

    print("weighted age:", weightedAgeBasis)
    if weightedAgeAvgLimit:
        print('\t weighted average age is {0}, -- {1}'.format(round(sum(result*weightedAgeAvg)/sum(result*ceu), 4), weightedAgeAvgLimit))
    if weightedAgeBasis:
        basis, _ = DecideBasis(weightedAgeBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if weightedAgeLimit[i]:
                print("\t weighted age from {0} to {1} is {2}, -- {3}:".format(weightedAgeLowBound[i], weightedAgeUpBound[i], round(sum(result*weightedAge[i]*basis)/sum(result*basis), 4), weightedAgeLimit[i]))    

    print("product:", productBasis)
    if productBasis:
        basis, _ = DecideBasis(productBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if productLimit[i]:
                print("\t product {0} is {1}, -- {2}:".format(productType[i], round(sum(result*product[i]*basis)/sum(result*basis), 4), productLimit[i]))    
    
    print("lessee:", lesseeBasis)
    if lesseeBasis:
        basis, _ = DecideBasis(lesseeBasis, var, ceu, teu, nbv, cost, queryID)
        print('\t Certain Lessee:')
        for i in range(numLimit):
            if lesseeLimit[i]:
                print("\t lessee {0} is {1}, -- {2}:".format(lesseeType[i], round(sum(result*lesseeOneHot[lesseeType[i]]*basis)/sum(result*basis), 4), lesseeLimit[i]))    
        print('\t Top lessee:')
        top3Lessee = heapq.nlargest(3, [(lesseeName, sum(result * lesseeOneHot[lesseeName])) for lesseeName in data['customer'].value_counts().index], key=lambda x:x[1])

        if topLesseeLimit[0]:
            print('\t top 1 {0} is {1}, -- {2}'.format(top3Lessee[0][0], top3Lessee[0][1]/sum(result*basis), topLesseeLimit[0]))
        if topLesseeLimit[1]:
            if len(top3Lessee) >= 2:
                print('\t top 2 {0} {1} is {2}, -- {3}'.format(top3Lessee[0][0], top3Lessee[1][0], (top3Lessee[0][1]+top3Lessee[1][1])/sum(result*basis), topLesseeLimit[1]))
            else:
                print('Only one lessee.')
        if topLesseeLimit[2]:
            if len(top3Lessee) >= 3:
                print('\t top 3 {0} {1} {2} is {3}, -- {4}'.format(top3Lessee[0][0], top3Lessee[1][0], top3Lessee[2][0], (top3Lessee[0][1]+top3Lessee[1][1]+top3Lessee[2][1])/sum(result*basis), topLesseeLimit[2]))
            else:
                print('Only two lessee.')
            
    print("contract type:", contractBasis)
    if contractBasis:
        basis, _ = DecideBasis(contractBasis, var, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if contractLimit[i]:
                print("\t contract type {0} is {1}, -- {2}:".format(contractType[i], round(sum(result*contract[i]*basis)/sum(result*basis), 4), contractLimit[i])) 



















