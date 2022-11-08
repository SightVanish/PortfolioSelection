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

numLimit = 5 # maximum num of constraints in each condition
timeLimit = 500
total_time = time.time()

if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)

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

def ConnectDatabase():
    """
    Load parameters in JSON from fll_t_dw.biz_fir_query_parameter_definition and load data from fll_t_dw.biz_ads_fir_pkg_data.
    """
    try:
        print('Parameters reading...')
        sqlParameter = "select python_json, id from fll_t_dw.biz_fir_query_parameter_definition where success_flag='T'"
        conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
        paramInput = pd.read_sql(sqlParameter, conn)
        if paramInput.shape[0] == 0:
            raise Exception('No Valid Query Request is Found!')
        elif paramInput.shape[0] > 1:
            raise Exception('More than One Valid Query Requests are Found!')
        queryID = paramInput['id'][0]
        param = json.loads(paramInput['python_json'][0])
    except Exception as e:
        print("Loading Parameters from GreenPlum Failed!\n", e)
        exit(1)

    try:
        print('Data loading...')
        print('Query ID:', queryID)
        sqlInput = """
            select billing_status_fz as billing, unit_id_fz as unit_id, product, fleet_year_fz as fleet_year, contract_cust_id as customer, \
            contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age, query_id, ceu_fz as ceu, teu_fz as teu
            from fll_t_dw.biz_ads_fir_pkg_data WHERE query_id='{0}'
        """.format(queryID) 
        data = pd.read_sql(sqlInput, conn)

        if data.shape[0] == 0:
            raise Exception("No Data Available!")
        print('Input data shape:', data.shape)
        print(param)
        conn.close()
    except Exception as e:
        print(e)
        ReportStatus("Loading Data from GreenPlum Failed!", 'F', queryID)
        exit(1)

    return queryID, param, data

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
        print("Writing data to GreenPlum Failed!\n", e) 
        ReportStatus("Writing data to GreenPlum Failed!", 'F', queryID)
        exit(1)


# queryID, param, data = ConnectDatabase()

# print('Data reading...')
# data = pd.read_csv('./local_data.csv')
# print('Data loading...')
# with open("./parameterDemo1.json") as f:
#     param = json.load(f)
# queryID = "local_test_id"
# print(param)
# print(data.shape)


print("==============================================================")
print('Parameters parsing...')
try:
    NbvCost = param['prefer']['nbvorCost']
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
    basis = {}
    basis['nbv'] = nbv
    basis['ceu'] = ceu
    basis['teu'] = teu
    basis['cost'] = cost
    hireStatus = {}
    hireStatus['ON'] = onHireStatus
    hireStatus['OF'] = offHireStatus
    hireStatus['None'] = noneHireStatus 
except Exception as e:
    print(e)
    ReportStatus('Processing Data Failed!', 'F', queryID)
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
                if lesseeType[i] in lesseeOneHot:
                    print('Set Lessee Limit', i)
                    if lesseeGeq[i]:
                        constraints.append(cp.sum(cp.multiply(x, lesseeOneHot[lesseeType[i]] * basis[lesseeBasis])) >= \
                            lesseeLimit[i] * cp.sum(cp.multiply(x, basis[lesseeBasis])))
                    else:
                        constraints.append(cp.sum(cp.multiply(x, lesseeOneHot[lesseeType[i]] * basis[lesseeBasis])) <= \
                            lesseeLimit[i] * cp.sum(cp.multiply(x, basis[lesseeBasis])))
                else:
                    print('Cannot Find', lesseeType[i])
        # top1
        for i in range(3):
            if topLesseeLimit[i]:
                print('Set Top', i+1)
                if topLesseeGeq[i]:
                    constraints.append(cp.sum_largest( \
                        cp.hstack([cp.sum(cp.multiply(x, lesseeOneHot[l] * basis[lesseeBasis])) for l in lesseeOneHot]), i+1) >= \
                            topLesseeLimit[0] * cp.sum(cp.multiply(x, basis[lesseeBasis])))
                else:
                    constraints.append(cp.sum_largest( \
                        cp.hstack([cp.sum(cp.multiply(x, lesseeOneHot[l] * basis[lesseeBasis])) for l in lesseeOneHot]), i+1) <= \
                            topLesseeLimit[0] * cp.sum(cp.multiply(x, basis[lesseeBasis])))  
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
                print('Set Produdct Limit', i)
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
    
    prob = cp.Problem(objective, constraints)
    print('Time Cost', time.time() - start_time)
    return prob, x



def SolveModel(prob, timeLimit):
    start_time = time.time()
    print("==============================================================")
    print('Model solving...')
    # solve model
    prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=timeLimit, numberThreads=4)
    print("==============================================================")
    print("status:", prob.status)
    print("==============================================================")
    print('Time Cost', time.time() - start_time)


prob, x = BuildModel()
prob = SolveModel(prob, timeLimit)



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
    if (maxTotalNbv or minTotalNbv) and passed:
        print('\t passed')
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
    if (maxTotalCost or minTotalCost) and passed:
        print('\t passed')

    print("container age:", fleetAgeBasis)
    if fleetAgeAvgLimit:
        resultFleetAgeAvg = sum(result*fleetAgeAvg)/sum(result)
        print('\t container average age is {0}'.format(round(resultFleetAgeAvg, 4)))
        if fleetAgeAvgGeq:
            if resultFleetAgeAvg < fleetAgeAvgLimit:
                passed = False
                print('\t \t >= failed')
        else:
            if resultFleetAgeAvg > fleetAgeAvgLimit:
                passed = False
                print('\t \t <= failed')
        if passed:
            print('\t \t passed')
    if fleetAgeBasis:
        resultFleetAge = [None for _ in range(numLimit)]
        for i in range(numLimit):
            if fleetAgeLimit[i]:
                resultFleetAge[i] = sum(result*fleetAge[i]*basis[fleetAgeBasis])/sum(result*basis[fleetAgeBasis])
                print("\t container age from {0} to {1}: {2}".format(fleetAgeLowBound[i], fleetAgeUpBound[i], round(resultFleetAge[i], 4)))
                if fleetAgeGeq[i]:
                    if resultFleetAge[i] < fleetAgeLimit[i]:
                        passed = False
                        print('\t \t >= failed')
                else:
                    if resultFleetAge[i] > fleetAgeLimit[i]:
                        passed = False
                        print('\t \t <= failed')
                if passed:
                    print('\t \t passed')

    print("weighted age:", weightedAgeBasis)
    if weightedAgeAvgLimit:
        resultWeightedAgeAvg = sum(result*weightedAgeAvg)/sum(result*ceu)
        print('\t weighted average age is {0}'.format(round(resultWeightedAgeAvg, 4)))
        if weightedAgeAvgGeq:
            if resultWeightedAgeAvg < weightedAgeAvgLimit:
                print('\t \t >= failed')
                passed = False
        else:
            if resultWeightedAgeAvg > weightedAgeAvgLimit:
                print('\t \t <= failed')
                passed = False
        if passed:
            print('\t \t passed')
    if weightedAgeBasis:
        resultWeightedAge = [None for _ in range(numLimit)]
        for i in range(numLimit):
            if weightedAgeLimit[i]:
                resultWeightedAge[i] = sum(result*weightedAge[i]*basis[weightedAgeBasis])/sum(result*basis[weightedAgeBasis])
                print("\t weighted age from {0} to {1} is {2}".format(weightedAgeLowBound[i], weightedAgeUpBound[i], round(resultWeightedAge[i], 4)))
                if weightedAgeGeq[i]:
                    if resultWeightedAge[i] < weightedAgeLimit[i]:
                        print('\t \t >= failed')
                        passed = False
                else:
                    if resultWeightedAge[i] > weightedAgeLimit[i]:
                        print('\t \t <= failed')
                        passed = False
                if passed:
                    print('\t \t passed')

    if lesseeBasis:
        print('Certain Lessee:', lesseeBasis)
        resultLessee = [None for _ in range(numLimit)]
        for i in range(numLimit):
            if lesseeLimit[i]:
                if lesseeType[i] in lesseeOneHot:
                    resultLessee[i] = sum(result*lesseeOneHot[lesseeType[i]]*basis[lesseeBasis])/sum(result*basis[lesseeBasis])
                    print("\t lessee {0} is {1}:".format(lesseeType[i], round(resultLessee[i], 4)))
                    if lesseeGeq[i]:
                        if resultLessee[i] < lesseeLimit[i]:
                            print('\t \t >= failed')
                            passed = False
                    else:
                        if resultLessee[i] > lesseeLimit[i]:
                            print('\t \t <= failed')
                            passed = False
                    if passed:
                        print('\t \t passed')
                else:
                    print('Can not find {0}'.format(lesseeType[i]))

        print('Top lessee:', lesseeBasis)
        top3Lessee = heapq.nlargest(3, [(lesseeName, sum(result*lesseeOneHot[lesseeName]*basis[lesseeBasis])) for lesseeName in data['customer'].value_counts().index], key=lambda x:x[1])
        resultTop3Lessee = [
            top3Lessee[0][1]/sum(result*basis[lesseeBasis]),
            (top3Lessee[0][1]+top3Lessee[1][1])/sum(result*basis[lesseeBasis]),
            (top3Lessee[0][1]+top3Lessee[1][1]+top3Lessee[2][1])/sum(result*basis[lesseeBasis])
        ]
        if topLesseeLimit[0]:
            print('\t top 1 {0} is {1}'.format(top3Lessee[0][0], round(resultTop3Lessee[0], 4)))
            if topLesseeGeq[0]:
                if resultTop3Lessee[0] < topLesseeLimit[0]:
                    print('\t \t >= failed')
                    passed = False
            else:
                if resultTop3Lessee[0] > topLesseeLimit[0]:
                    print('\t \t <= failed')
                    passed = False
            if passed:
                print('\t \t passed')
        if topLesseeLimit[1]:
            if len(top3Lessee) >= 2:
                print('\t top 2 {0} {1} is {2}'.format(top3Lessee[0][0], top3Lessee[1][0], round(resultTop3Lessee[1], 4)))
                if topLesseeGeq[1]:
                    if resultTop3Lessee[1] < topLesseeLimit[1]:
                        print('\t \t >= failed')
                        passed = False
                    else:
                        if resultTop3Lessee[1] > topLesseeLimit[1]:
                            print('\t \t <= failed')
                            passed = False
                if passed:
                    print('\t \t passed')
            else:
                print('\t Only one lessee.')
        if topLesseeLimit[2]:
            if len(top3Lessee) >= 3:
                print('\t top 3 {0} {1} {2} is {3}'.format(top3Lessee[0][0], top3Lessee[1][0], top3Lessee[2][0], round(resultTop3Lessee[2], 4)))
                if topLesseeGeq[2]:
                    if resultTop3Lessee[2] < topLesseeLimit[2]:
                        print('\t \t >= failed')
                        passed = False
                    else:
                        if resultTop3Lessee[2] > topLesseeLimit[2]:
                            print('\t \t <= failed')
                            passed = False
                if passed:
                    print('\t \t passed')
            else:
                print('\t Only two lessee.')
        
    print('billing status:', statusBasis)
    if statusBasis:
        resultStatus = [None for _ in range(numLimit)]
        for i in range(numLimit):
            if statusType[i]:
                if statusType[i] == 'ON':
                    resultStatus[i] = sum(result*onHireStatus*basis[statusBasis])/sum(result*basis[statusBasis])
                    print('\t OnHire is {0}'.format(round(resultStatus[i], 4)))
                if statusType[i] == 'OF':
                    resultStatus[i] = sum(result*offHireStatus*basis)/sum(result*basis)
                    print('\t OffHire is {0}'.format(round(resultStatus[i], 4)))
                if statusType[i] == 'None':
                    resultStatus[i] = sum(result*noneHireStatus*basis)/sum(result*basis)
                    print('\t NoneHire is {0}'.format(round(resultStatus[i], 4)))
                
                if statusGeq[i]:
                    if resultStatus[i] < statusLimit[i]:
                        print('\t \t >= failed')
                        passed = False
                else:
                    if resultStatus[i] > statusLimit[i]:
                        print('\t \t <= failed')
                        passed = False
                if passed:
                    print('\t \t passed')

    print("product:", productBasis)
    if productBasis:
        resultProduct = [None for _ in range(numLimit)]
        for i in range(numLimit):
            if productLimit[i]:
                resultProduct[i] = sum(result*product[i]*basis[productBasis])/sum(result*basis[productBasis])
                print("\t product {0} is {1}:".format(productType[i], round(resultProduct[i], 4)))
                if productGeq[i]:
                    if resultProduct[i] < productLimit[i]:
                        print('\t \t >= failed')
                        passed = False
                else:
                    if resultProduct[i] > productLimit[i]:
                        print('\t \t <= failed')
                        passed = False
                if passed:
                    print('\t \t passed')


    print("contract type:", contractBasis)
    if contractBasis:
        resultContract = [None for _ in range(numLimit)]
        for i in range(numLimit):
            if contractLimit[i]:
                resultContract[i] = sum(result*contract[i]*basis[contractBasis])/sum(result*basis[contractBasis])
                print("\t contract type {0} is {1}:".format(contractType[i], round(resultContract[i], 4))) 
                if contractGeq[i]:
                    if resultContract[i] < contractLimit[i]:
                        print('\t \t >= failed')
                else:
                    if resultContract[i] > contractLimit[i]:
                        print('\t \t <= failed')
                if passed:
                    print('\t \t passed')

    if passed:
        print('Algorithm Succeeded!!!!!!!!!!!!!!!!')
    return passed

