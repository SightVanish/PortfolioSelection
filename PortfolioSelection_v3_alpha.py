import numpy as np
import pandas as pd
import time
import heapq
import json
import psycopg2
import psycopg2.extras
import warnings
import sys
import uuid
import cvxpy as cp
import datetime
import argparse
"""
Check before submit
1. enable queryID test
2. switch database 
3. bug free
"""
INF = float('inf')
total_time = time.time()
local = False # use local file or not

if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)

parser = argparse.ArgumentParser(description="This is Flornes Porfolio Selection Model.")
parser.add_argument('--queryID', '-id', type=str, help='Query ID')
parser.add_argument('--numLimit', '-n', type=int, default=5, help='Maximum number of constraints in each condition')
parser.add_argument('--threadLimit', '-t', type=int, default=4, help='Maximum number of threads')
args = parser.parse_args().__dict__
numLimit = args['numLimit']
threadLimit = args['threadLimit']
queryID = args['queryID']
print('Input argparse',  args)

if queryID is None:
    print("No valid query id!")
    exit(1)

def ReportStatus(msg, flag, queryID):
    """
    Print message and update status in fll_t_dw.biz_fir_query_parameter_definition.
    """
    sql = "update fll_t_dw.biz_fir_query_parameter_definition set python_info_data='{0}', success_flag='{1}', update_time='{2}' where id='{3}'".format(msg, flag, datetime.datetime.now(), queryID)
    print("============================================================================================================================")
    print("Reporting issue:", msg)
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    conn.close()

def ConnectDatabase(queryID):
    """
    Load parameters in JSON from fll_t_dw.biz_fir_query_parameter_definition and load data from fll_t_dw.biz_ads_fir_pkg_data.
    """
    # load parameters
    try:
        print('Parameters reading...')
        sqlParameter = "select python_json from fll_t_dw.biz_fir_query_parameter_definition where id='{0}'".format(queryID)
        conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
        paramInput = pd.read_sql(sqlParameter, conn)
        if paramInput.shape[0] == 0:
            raise Exception("No Valid Query Request is Found!")
        elif paramInput.shape[0] > 1:
            raise Exception("More than One Valid Query Requests are Found!")
        param = json.loads(paramInput['python_json'][0])
        print(param)
    except Exception as e:
        print("Loading Parameters from GreenPlum Failed!\n", e)
        exit(1)
    
    # load data
    try:
        print('Data loading...')
        sqlInput = """
            select billing_status_fz as billing, unit_id_fz as unit_id, product, fleet_year_fz as fleet_year, contract_cust_id as customer, \
            contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age, query_id, ceu_fz as ceu, teu_fz as teu, rent as rent, rml_x_ceu as rml
            from fll_t_dw.biz_ads_fir_pkg_data WHERE query_id='{0}'
        """.format(queryID) 
        data = pd.read_sql(sqlInput, conn)
        if data.shape[0] == 0:
            raise Exception("No Data Available!")
        print('Input data shape:', data.shape)
        conn.close()
    except Exception as e:
        print(e)
        ReportStatus("Loading Data from GreenPlum Failed!", 'F', queryID)
        exit(1)

    return param, data

def OutputPackage(data, result, queryID):
    """
    Output final package to fll_t_dw.biz_fir_asset_package.
    """
    sqlOutput = "insert into fll_t_dw.biz_fir_asset_package (unit_id, query_id, id, is_void, version) values %s"
    try:
        conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
        conn.autocommit = True
        cur = conn.cursor()
        print('Writing data...')
        values_list = []
        for i in range(len(result)):
            if result[i]:
                values_list.append((data['unit_id'][i], queryID, uuid.uuid1().hex, 0, 0))
        psycopg2.extras.execute_values(cur, sqlOutput, values_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e) 
        ReportStatus("Writing data to GreenPlum Failed!", 'F', queryID)
        exit(1)



if local:
    print('Data reading...')
    data = pd.read_csv('./local_data.csv')
    print('Parameter loading...')
    with open("./parameterDemoTest.json") as f:
        param = json.load(f)
    queryID = "local_test_id"
    print("==============================================================")
    print(param)
    print(data.shape)
else:
    param, data = ConnectDatabase(queryID)

print("==============================================================")
print('Parameters parsing...')
try:
    timeLimit = param['timeLimit'] if param['timeLimit'] > 0 else 600
    print('model time limit:', timeLimit)
    NbvCost = param['prefer']['nbvorCost']
    maxOrMin = param['prefer']['maxOrMin']
    fleetAgeLowBound = [-INF for _ in range(numLimit)]
    fleetAgeUpBound = [INF for _ in range(numLimit)]
    fleetAgeLimit = [None for _ in range(numLimit)]
    fleetAgeGeq = [None for _ in range(numLimit)]
    weightedAgeLowBound = [-INF for _ in range(numLimit)]
    weightedAgeUpBound = [INF for _ in range(numLimit)]
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
    rmlLowBound = [-INF for _ in range(numLimit)]
    rmlUpBound = [INF for _ in range(numLimit)]
    rmlGeq = [None for _ in range(numLimit)]
    rmlLimit = [None for _ in range(numLimit)]

    minTotalNbv = param['totalNBVFrom']
    maxTotalNbv = param['totalNBVTo']

    minTotalCost = param['totalCostFrom']
    maxTotalCost = param['totalCostTo']

    minTotalRent = param['totalRentFrom']

    lesseeOthers = param['lessee']['others']['lessee']
    lesseeOthersLimit = param['lessee']['others']['percent'] / 100

    topLesseeLimit = [
        param['lessee']['topLessee']['top1']['percent'] / 100,
        param['lessee']['topLessee']['top2']['percent'] / 100,
        param['lessee']['topLessee']['top3']['percent'] / 100]
    topLesseeGeq = [
        param['lessee']['topLessee']['top1']['symbol'],
        param['lessee']['topLessee']['top2']['symbol'],
        param['lessee']['topLessee']['top3']['symbol']]

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

    rmlBasis = param['rml']['basis']
    for i in range(len(param['rml']['list'])):
        rmlLowBound[i] = param['rml']['list'][i]['rmlFrom']
        # rmlUpBound[i] = param['rml']['list'][i]['rmlTo']
        rmlGeq[i] = param['rml']['list'][i]['symbol']
        rmlLimit[i] = param['rml']['list'][i]['percent'] / 100
except Exception as e:
    print(e)
    msg = 'Parsing Paramters Failed! ' + str(e)
    ReportStatus(msg, 'F', queryID)
    exit(1)


print("==============================================================")
print('Data processing...')
try:
    # Billing Status
    data['OnHireStatus'] = data['billing'].apply(lambda x: 1 if x=='ON' else 0)
    data['OffHireStatus'] = data['billing'].apply(lambda x: 1 if x=='OF' else 0)
    data['NoneStatus'] = data['billing'].apply(lambda x: 1 if (x!='ON' and x!='OF') else 0)
    # One hot all lessees
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
        # RML
        if rmlLimit[i]:
            column_name = 'RML{0}'.format(i)
            data[column_name] = data['rml'].apply(lambda x: 1 if rmlLowBound[i]<=x<=rmlUpBound[i] else 0)
    
    # convert data to numpy
    nbv = data['nbv'].to_numpy()
    cost = data['cost'].to_numpy()
    ceu = data['ceu'].to_numpy()
    teu = data['teu'].to_numpy()
    rent = data['rent'].to_numpy()
    fleetAgeAvg = data['fleet_year'].to_numpy()
    weightedAgeAvg = data['weighted_age'].to_numpy()
    onHireStatus = data['OnHireStatus'].to_numpy()
    offHireStatus = data['OffHireStatus'].to_numpy()
    noneHireStatus = data['NoneStatus'].to_numpy()
    hireStatus = {}
    hireStatus['ON'] = onHireStatus
    hireStatus['OF'] = offHireStatus
    hireStatus['None'] = noneHireStatus 
    lesseeOneHot = {lesseeName: data[lesseeName].to_numpy() for lesseeName in data['customer'].value_counts().index}
    fleetAge = []
    weightedAge = []
    product = []
    contract = []
    rml = []
    for i in range(numLimit):
        fleetAge.append(data['FleetAge{0}'.format(i)].to_numpy() if fleetAgeLimit[i] else None)
        weightedAge.append(data['WeightedAge{0}'.format(i)].to_numpy() if weightedAgeLimit[i] else None)
        product.append(data['ProductType{0}'.format(i)].to_numpy() if productLimit[i] else None)
        contract.append(data['ContractType{0}'.format(i)].to_numpy() if contractLimit[i] else None)
        rml.append(data['RML{0}'.format(i)].to_numpy() if rmlLimit[i] else None)
    basis = {}
    basis['nbv'] = nbv
    basis['ceu'] = ceu
    basis['teu'] = teu
    basis['cost'] = cost
except Exception as e:
    print(e)
    msg = 'Processing Data Failed!' + str(e)
    ReportStatus(msg, 'F', queryID)
    exit(1)


def BuildModel():
    print("==============================================================")
    print('Model preparing...')
    start_time = time.time()

    x = cp.Variable(shape=data.shape[0], boolean=True)
    # objective function 
    if NbvCost:
        obj = cp.sum(cp.multiply(x, nbv))
    else:
        obj = cp.sum(cp.multiply(x, cost))
    if maxOrMin:
        objective = cp.Maximize(obj)
    else:
        objective = cp.Minimize(obj)

    # constraints
    constraints = []
    # nbv
    if maxTotalNbv:
        constraints.append(cp.sum(cp.multiply(x, nbv)) <= maxTotalNbv)
        print('Set Max Nbv')
    if minTotalNbv:
        constraints.append(cp.sum(cp.multiply(x, nbv)) >= minTotalNbv)
        print('Set Min Nbv')
    # cost
    if maxTotalCost:
        constraints.append(cp.sum(cp.multiply(x, cost)) <= maxTotalCost)
        print('Set Max Cost')
    if minTotalCost:
        constraints.append(cp.sum(cp.multiply(x, cost)) >= minTotalCost)
        print('Set Min Cost')
    # rent
    if minTotalRent:
        constraints.append(cp.sum(cp.multiply(x, rent)) >= minTotalRent)
    # container age
    if fleetAgeAvgLimit:
        print('Set Container Average Age Limit')
        if fleetAgeAvgGeq:
            constraints.append(cp.sum(cp.multiply(x, fleetAgeAvg)) >= fleetAgeAvgLimit * cp.sum(x))
        else:
            constraints.append(cp.sum(cp.multiply(x, fleetAgeAvg)) <= fleetAgeAvgLimit * cp.sum(x))
    if fleetAgeBasis:
        for i in range(numLimit):
            if fleetAgeLimit[i]:
                print('Set Container Age Limit', i)
                if fleetAgeGeq[i]:
                    constraints.append(cp.sum(cp.multiply(x, fleetAge[i] * basis[fleetAgeBasis])) >= \
                        fleetAgeLimit[i] * cp.sum(cp.multiply(x, basis[fleetAgeBasis])))
                else:
                    constraints.append(cp.sum(cp.multiply(x, fleetAge[i] * basis[fleetAgeBasis])) <= \
                        fleetAgeLimit[i] * cp.sum(cp.multiply(x, basis[fleetAgeBasis])))
    # weighted age
    if weightedAgeAvgLimit:
        print('Set Weighted Average Age Limit')
        if weightedAgeAvgGeq:
            constraints.append(cp.sum(cp.multiply(x, weightedAgeAvg)) >= \
                weightedAgeAvgLimit * cp.sum(cp.multiply(x, ceu)))
        else:
            constraints.append(cp.sum(cp.multiply(x, weightedAgeAvg)) <= \
                weightedAgeAvgLimit * cp.sum(cp.multiply(x, ceu)))
    if weightedAgeBasis:
        for i in range(numLimit):
            if weightedAgeLimit[i]:
                print('Set Weighted Age Limit', i)
                if weightedAgeGeq[i]:
                    constraints.append(cp.sum(cp.multiply(x, weightedAge[i] * basis[weightedAgeBasis])) >= \
                        weightedAgeLimit[i] * cp.sum(cp.multiply(x, basis[weightedAgeBasis])))
                else:
                    constraints.append(cp.sum(cp.multiply(x, weightedAge[i] * basis[weightedAgeBasis])) <= \
                        weightedAgeLimit[i] * cp.sum(cp.multiply(x, basis[weightedAgeBasis])))
    # lessee
    if lesseeBasis:
        for i in range(numLimit):
            if lesseeLimit[i]:
                if lesseeType[i] not in lesseeOneHot:
                    lesseeOneHot[lesseeType[i]] = np.zeros(data.shape[0])
                print('Set Lessee Limit', i)
                if lesseeGeq[i]:
                    constraints.append(cp.sum(cp.multiply(x, lesseeOneHot[lesseeType[i]] * basis[lesseeBasis])) >= \
                        lesseeLimit[i] * cp.sum(cp.multiply(x, basis[lesseeBasis])))
                else:
                    constraints.append(cp.sum(cp.multiply(x, lesseeOneHot[lesseeType[i]] * basis[lesseeBasis])) <= \
                        lesseeLimit[i] * cp.sum(cp.multiply(x, basis[lesseeBasis])))
        maxTop = 0
        # top lessee
        for i in range(3):
            if topLesseeLimit[i]:
                maxTop = i+1
                print('Set Top', i+1)
                if topLesseeGeq[i]:
                    constraints.append(cp.sum_largest( \
                        cp.hstack([cp.sum(cp.multiply(x, lesseeOneHot[l] * basis[lesseeBasis])) for l in lesseeOneHot]), i+1) >= \
                            topLesseeLimit[i] * cp.sum(cp.multiply(x, basis[lesseeBasis])))
                else:
                    constraints.append(cp.sum_largest( \
                        cp.hstack([cp.sum(cp.multiply(x, lesseeOneHot[l] * basis[lesseeBasis])) for l in lesseeOneHot]), i+1) <= \
                            topLesseeLimit[i] * cp.sum(cp.multiply(x, basis[lesseeBasis])))
        # others
        if lesseeOthersLimit:
            if lesseeOthers:
                print('Set Other Lessees via List')
                # add constraints according to user input
                constraints.append(cp.sum_largest( \
                    cp.hstack([cp.sum(cp.multiply(x, lesseeOneHot[l] * basis[lesseeBasis])) for l in lesseeOthers]), 1) <= \
                        lesseeOthersLimit * cp.sum(cp.multiply(x, basis[lesseeBasis])))
            else:
                # find max top limit
                print('Set Other Lessee')
                print('\t Max Top:', maxTop)
                constraints.append(
                    cp.sum_largest(cp.hstack([cp.sum(cp.multiply(x, lesseeOneHot[l] * basis[lesseeBasis])) for l in lesseeOneHot]), maxTop + 1) <= \
                        lesseeOthersLimit * cp.sum(cp.multiply(x, basis[lesseeBasis]))) + cp.sum_largest(cp.hstack([cp.sum(cp.multiply(x, lesseeOneHot[l] * basis[lesseeBasis])) for l in lesseeOneHot]), maxTop)
        
    # status
    if statusBasis:
        for i in range(numLimit):
            if statusType[i]:
                print('Set Status Limit', i)
                if statusGeq[i]:
                    constraints.append(cp.sum(cp.multiply(x, hireStatus[statusType[i]] * basis[statusBasis])) >= \
                        statusLimit[i] * cp.sum(cp.multiply(x, basis[statusBasis])))
                else:
                    constraints.append(cp.sum(cp.multiply(x, hireStatus[statusType[i]] * basis[statusBasis])) <= \
                        statusLimit[i] * cp.sum(cp.multiply(x, basis[statusBasis])))
    # product
    if productBasis:
        for i in range(numLimit):
            if productLimit[i]:
                print('Set Product Limit', i)
                if productGeq[i]:
                    constraints.append(cp.sum(cp.multiply(x, product[i] * basis[productBasis])) >= \
                        productLimit[i] * cp.sum(cp.multiply(x, basis[productBasis])))
                else:
                    constraints.append(cp.sum(cp.multiply(x, product[i] * basis[productBasis])) <= \
                        productLimit[i] * cp.sum(cp.multiply(x, basis[productBasis])))
    # contract type
    if contractBasis:
        for i in range(numLimit):
            if contractLimit[i]:
                print('Set Contract Type Limit', i)
                if contractGeq[i]:
                    constraints.append(cp.sum(cp.multiply(x, contract[i] * basis[contractBasis])) >= \
                        contractLimit[i] * cp.sum(cp.multiply(x, basis[contractBasis])))
                else:
                    constraints.append(cp.sum(cp.multiply(x, contract[i] * basis[contractBasis])) <= \
                        contractLimit[i] * cp.sum(cp.multiply(x, basis[contractBasis])))
    # rml
    if rmlBasis:
        for i in range(numLimit):
            if rmlLimit[i]:
                print('Set RML limit', i)
                if rmlGeq[i]:
                    constraints.append(cp.sum(cp.multiply(x, rml[i] * basis[rmlBasis])) >= \
                        rmlLimit[i] * cp.sum(cp.multiply(x, basis[rmlBasis])))
                else:
                    constraints.append(cp.sum(cp.multiply(x, rml[i] * basis[rmlBasis])) <= \
                        rmlLimit[i] * cp.sum(cp.multiply(x, basis[rmlBasis])))
    
    prob = cp.Problem(objective, constraints)
    print('Time Cost', time.time() - start_time)
    return prob, x

def SolveModel(prob, timeLimit, threadLimit):
    start_time = time.time()
    print("==============================================================")
    print('Model solving...')
    # solve model
    prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=timeLimit, numberThreads=threadLimit)
    print("==============================================================")
    print("status:", prob.status)
    print("==============================================================")
    print('Time Cost', time.time() - start_time)
    return prob

try:
    prob, x = BuildModel()
    prob = SolveModel(prob, timeLimit, threadLimit)
except Exception as e:
    print(e)
    ReportStatus('Model Failed! Please Contact Developing Team!', 'F', queryID)
    exit(1)


def ValidResult(result):
    passed = True
    print('======================================================================')
    resultNbv = sum(result*nbv)
    print("nbv: {0}".format(round(resultNbv, 4)))
    if maxTotalNbv:
        if (resultNbv - maxTotalNbv) > 0.1:
            passed = False
            print('\t max failed')
    if minTotalNbv:
        if (minTotalNbv - resultNbv) > 0.1: 
            passed = False
            print('\t min failed')
    resultCost = sum(result*cost)
    print("cost: {0}".format(round(resultCost, 4)))
    if maxTotalCost:
        if (resultCost - maxTotalCost) > 0.1:
            passed = False
            print('\t max failed')
    if minTotalCost:
        if (minTotalCost - resultCost) > 0.1:
            passed = False
            print('\t min failed')
    resultRent = sum(result*rent)
    print("rent: {0}".format(round(resultRent, 4)))
    if minTotalRent:
        if (minTotalCost - resultRent) > 0.1:
            passed = False
            print('\t min failed')

    if fleetAgeAvgLimit:
        resultFleetAgeAvg = sum(result*fleetAgeAvg)/sum(result)
        print('container average age is {0}'.format(round(resultFleetAgeAvg, 4)))
        if fleetAgeAvgGeq:
            if resultFleetAgeAvg < fleetAgeAvgLimit:
                passed = False
                print('\t >= failed')
        else:
            if resultFleetAgeAvg > fleetAgeAvgLimit:
                passed = False
                print('\t <= failed')

    if fleetAgeBasis:
        for i in range(numLimit):
            if fleetAgeLimit[i]:
                resultFleetAge = sum(result*fleetAge[i]*basis[fleetAgeBasis])/sum(result*basis[fleetAgeBasis])
                print("container age from {0} to {1} is {2}".format(fleetAgeLowBound[i], fleetAgeUpBound[i], round(resultFleetAge, 4)))
                if fleetAgeGeq[i]:
                    if resultFleetAge < fleetAgeLimit[i]:
                        passed = False
                        print('\t >= failed')
                else:
                    if resultFleetAge > fleetAgeLimit[i]:
                        passed = False
                        print('\t <= failed')

    if weightedAgeAvgLimit:
        resultWeightedAgeAvg = sum(result*weightedAgeAvg)/sum(result*ceu)
        print('weighted average age is {0}'.format(round(resultWeightedAgeAvg, 4)))
        if weightedAgeAvgGeq:
            if resultWeightedAgeAvg < weightedAgeAvgLimit:
                print('\t >= failed')
                passed = False
        else:
            if resultWeightedAgeAvg > weightedAgeAvgLimit:
                print('\t <= failed')
                passed = False

    if weightedAgeBasis:
        for i in range(numLimit):
            if weightedAgeLimit[i]:
                resultWeightedAge = sum(result*weightedAge[i]*basis[weightedAgeBasis])/sum(result*basis[weightedAgeBasis])
                print("weighted age from {0} to {1} is {2}".format(weightedAgeLowBound[i], weightedAgeUpBound[i], round(resultWeightedAge, 4)))
                if weightedAgeGeq[i]:
                    if resultWeightedAge < weightedAgeLimit[i]:
                        print('\t >= failed')
                        passed = False
                else:
                    if resultWeightedAge > weightedAgeLimit[i]:
                        print('\t <= failed')
                        passed = False

    if lesseeBasis:
        for i in range(numLimit):
            if lesseeLimit[i]:
                resultLessee = sum(result*lesseeOneHot[lesseeType[i]]*basis[lesseeBasis])/sum(result*basis[lesseeBasis])
                print("lessee {0} is {1}".format(lesseeType[i], round(resultLessee, 4)))
                if lesseeGeq[i]:
                    if resultLessee < lesseeLimit[i]:
                        print('\t >= failed')
                        passed = False
                else:
                    if resultLessee > lesseeLimit[i]:
                        print('\t <= failed')
                        passed = False

        top4Lessee = heapq.nlargest(4, [(lesseeName, sum(result*lesseeOneHot[lesseeName]*basis[lesseeBasis])) for lesseeName in data['customer'].value_counts().index], key=lambda x:x[1])
        resultTop3Lessee = [
            sum(i[1] for i in top4Lessee[:1]) / sum(result*basis[lesseeBasis]),
            sum(i[1] for i in top4Lessee[:2]) / sum(result*basis[lesseeBasis]),
            sum(i[1] for i in top4Lessee[:3]) / sum(result*basis[lesseeBasis])
        ]
        for i in range(3):
            if topLesseeLimit[i]:
                print('top {0} {1} is {2}'.format(i+1, [i[0] for i in top4Lessee[:i+1]], round(resultTop3Lessee[i], 4)))
                if topLesseeGeq[i]:
                    if resultTop3Lessee[i] < topLesseeLimit[i]:
                        print('\t >= failed')
                        passed = False
                else:
                    if resultTop3Lessee[i] > topLesseeLimit[i]:
                        print('\t <= failed')
                        passed = False
        # others
        if lesseeOthersLimit:
            if lesseeOthers:
                print('Other lessees via list')
                otherLessees = [sum(result*lesseeOneHot[l]*basis[lesseeBasis]) for l in lesseeOthers]
                print('\t top others is {0}'.format(round(max(otherLessees), 4)))
                if max(otherLessees) > lesseeOthersLimit:
                    print('\t \t failed')
                    passed = False
            else:
                print('Other lesees')
                maxTop = 0
                for i in range(3):
                    if topLesseeLimit[i]:
                        maxTop = i+1
                print('\t top others is {0}'.format(round(top4Lessee[maxTop][1]/sum(result*basis[lesseeBasis]), 4)))
                if top4Lessee[maxTop][1]/sum(result*basis[lesseeBasis]) > lesseeOthersLimit:
                    print('\t \t failed')
                    passed = False

    if statusBasis:
        for i in range(numLimit):
            if statusType[i]:
                resultStatus = sum(result*hireStatus[statusType[i]] *basis[statusBasis])/sum(result*basis[statusBasis])
                print('Hire {0} is {1}'.format(statusType[i], round(resultStatus, 4)))
                if statusGeq[i]:
                    if resultStatus < statusLimit[i]:
                        print('\t >= failed')
                        passed = False
                else:
                    if resultStatus > statusLimit[i]:
                        print('\t <= failed')
                        passed = False

    if productBasis:
        for i in range(numLimit):
            if productLimit[i]:
                resultProduct = sum(result*product[i]*basis[productBasis])/sum(result*basis[productBasis])
                print("product {0} is {1}".format(productType[i], round(resultProduct, 4)))
                if productGeq[i]:
                    if resultProduct < productLimit[i]:
                        print('\t >= failed')
                        passed = False
                else:
                    if resultProduct > productLimit[i]:
                        print('\t <= failed')
                        passed = False

    if contractBasis:
        for i in range(numLimit):
            if contractLimit[i]:
                resultContract = sum(result*contract[i]*basis[contractBasis])/sum(result*basis[contractBasis])
                print("contract type {0} is {1}".format(contractType[i], round(resultContract, 4))) 
                if contractGeq[i]:
                    if resultContract < contractLimit[i]:
                        print('\t >= failed')
                        passed = False
                else:
                    if resultContract > contractLimit[i]:
                        print('\t <= failed')
                        passed = False
    if rmlBasis:
        for i in range(numLimit):
            resultRML = sum(result*rml[i]*basis[rmlBasis])/sum(result*basis[rmlBasis])
            print("rml from {0} to {1} is {2}".format(rmlLowBound[i], rmlUpBound[i], round(resultRML, 4)))
            if rmlGeq[i]:
                if resultRML < rmlLimit[i]:
                    print('\t >= failed')
                    passed = False
            else:
                if resultRML > rmlLimit[i]:
                    print('\t <= failed')
                    passed = False

    if passed:
        print('Algorithm Succeeded!!!!!!!!!!!!!!!!')
    return passed

if prob.status == 'infeasible':
    ReportStatus('Problem Proven Infeasible! Please Modify Constaints.', 'I', queryID)
else:
    try:
        result = x.value
        print('Result is Valid:', len(set(result)) == 2)
        result = np.where(abs(result-1) < 1e-3, 1, 0) # x == 1
        print(int(sum(result)), '/', len(result), 'containers are selected.')

        if int(sum(result)) == 0:
            ReportStatus('Constraints Cannot Be fulfilled! Please Modify Constaints.', 'I', queryID)
        else:
            passed = ValidResult(result)
            OutputPackage(data, result, queryID)
            if passed:
                ReportStatus('Algorithm Succeeded!', 'O', queryID)
            else:
                ReportStatus('Constraints Cannot Be fulfilled! Please Modify Constaints Or Increase Running Timelimit.', 'N', queryID)
    except Exception as e:
        print(e)
        ReportStatus('Result Validation Failed!', 'F', queryID)
        exit(1)

print('Total Time Cost:', time.time() - total_time)



