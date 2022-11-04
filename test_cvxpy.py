import numpy as np
import pandas as pd
import time
import json
import psycopg2
import psycopg2.extras
import warnings
import sys
import uuid
import cvxpy as cp

numLimit = 5 # maximum num of constraints in each condition
timeLimit = 200
total_time = time.time()

if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)

def ReportStatus(msg, flag, queryID):
    """
    Print message and update status in fll_t_dw.biz_fir_query_parameter_definition.
    """
    sql = "update fll_t_dw.biz_fir_query_parameter_definition set python_info_data='{0}', success_flag='{1}' where id='{2}'".format(msg, flag, queryID)
    print("============================================================================================================================")
    print("Reporting issue:", msg)
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    conn.close()

def DecideBasis(basis, ceu, teu, nbv, cost, queryID):
    """
    Return [ceu / teu / nbv / cost] basde on basis.
    """
    if basis == 'ceu':
        return ceu
    elif basis == 'teu':
        return ceu
    elif basis == 'nbv':
        return ceu
    elif basis == 'cost':
        return ceu
    else:
        ReportStatus('Basis is not valid!', 'F', queryID)

def DecideStatus(status, onHireStatus, offHireStatus, noneHireStatus, queryID):
    """
    Return [OnHire / OffHire / None] based on status.
    """
    if status == 'ON':
        return "OnHire", onHireStatus
    elif status == 'OF':
        return "OffHire", offHireStatus
    elif status == 'None':
        return "NoneHire", noneHireStatus
    else:
        ReportStatus('Status is not valid!', 'F', queryID)

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
    sqlOutput = "insert into fll_t_dw.biz_fir_asset_package (unit_id, query_id, id) values %s"
    try:
        conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")
        conn.autocommit = True
        cur = conn.cursor()
        print('Writing data...')
        values_list = []
        for i in range(len(result)):
            if result[i]:
                values_list.append((data['unit_id'][i], queryID, uuid.uuid1().hex))
        psycopg2.extras.execute_values(cur, sqlOutput, values_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print("Writing data to GreenPlum Failed!\n", e) 
        ReportStatus("Writing data to GreenPlum Failed!", 'F', queryID)
        exit(1)


print('Data reading...')
rawData = pd.read_excel(io='./test_data_with_constraints.xlsb', \
    sheet_name='数据', engine='pyxlsb')
rawData = rawData[['Unit Id Fz', 'Contract Num', 'Cost', 'Product', \
    'Contract Cust Id', 'Contract Lease Type', 'Nbv', 'Billing Status Fz', \
    'Fleet Year Fz', 'Age x CEU', 'Ceu Fz', 'Teu Fz']].copy()
rawData.columns = ['unit_id', 'contract_num', 'cost', 'product', \
    'customer', 'contract', 'nbv', 'billing', \
    'fleet_year', 'weighted_age', 'ceu', 'teu']

print('Data loading...')
with open("./parameterDemo1.json") as f:
    param = json.load(f)
data = rawData.copy()
queryID = "local_test_id"
print(param)
print(data.shape)



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

def BuildModel():
    print("==============================================================")
    print('Model preparing...')

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
        basis = DecideBasis(fleetAgeBasis, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if fleetAgeLimit[i]:
                print('Set Container Age Limit', i)
                if fleetAgeGeq[i]:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, fleetAge[i]), basis)) <= \
                        fleetAgeLimit[i] * cp.sum(cp.multiply(x, basis)))
                else:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, fleetAge[i]), basis)) >= \
                        fleetAgeLimit[i] * cp.sum(cp.multiply(x, basis)))
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
        basis = DecideBasis(weightedAgeBasis, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if weightedAgeLimit[i]:
                print('Set Weighted Age Limit', i)
                if weightedAgeGeq[i]:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, weightedAge[i]), basis)) >= \
                        weightedAgeLimit[i] * cp.sum(cp.multiply(x, basis)))

                else:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, weightedAge[i]), basis)) <= \
                        weightedAgeLimit[i] * cp.sum(cp.multiply(x, basis)))

    # lessee
    if lesseeBasis:
        basis = DecideBasis(lesseeBasis, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if lesseeLimit[i]:
                if lesseeType[i] in lesseeOneHot:
                    print('Set Lessee Limit', i)
                    if lesseeGeq[i]:
                        constraints.append(cp.sum(cp.multiply(cp.multiply(x, lesseeOneHot[lesseeType[i]]), \
                            basis)) >= lesseeLimit[i] * cp.sum(cp.multiply(x, basis)))
                    else:
                        constraints.append(cp.sum(cp.multiply(cp.multiply(x, lesseeOneHot[lesseeType[i]]), \
                            basis)) <= lesseeLimit[i] * cp.sum(cp.multiply(x, basis)))
                else:
                    print('Cannot Find', lesseeType[i])
        # top1
        constraints.append(cp.sum_largest( \
            cp.hstack([cp.sum(cp.multiply(cp.multiply(x, lesseeOneHot[i]), basis)) for i in lesseeOneHot]), 1) <= \
                topLesseeLimit[0] * cp.sum(cp.multiply(x, basis)))
        # top3
        constraints.append(cp.sum_largest( \
            cp.hstack([cp.sum(cp.multiply(cp.multiply(x, lesseeOneHot[i]), basis)) for i in lesseeOneHot]), 3) <= \
                topLesseeLimit[2] * cp.sum(cp.multiply(x, basis)))
    



    
    # status
    if statusBasis:
        basis = DecideBasis(statusBasis, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if statusType[i]:
                print('Set Status Limit', i)
                statusName, status = DecideStatus(statusType[i], onHireStatus, offHireStatus, noneHireStatus, queryID)
                if statusGeq[i]:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, status), basis)) >= \
                        statusLimit[i] * cp.sum(cp.multiply(x, basis)))
                else:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, status), basis)) <= \
                        statusLimit[i] * cp.sum(cp.multiply(x, basis)))
    # product
    if productBasis:
        basis = DecideBasis(productBasis, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if productLimit[i]:
                print('Set Produdct Limit', i)
                if productGeq[i]:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, product[i]), basis)) >= \
                        productLimit[i] * cp.sum(cp.multiply(x, basis)))
                else:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, product[i]), basis)) <= \
                        productLimit[i] * cp.sum(cp.multiply(x, basis)))
    # contract type
    if contractBasis:
        basis = DecideBasis(contractBasis, ceu, teu, nbv, cost, queryID)
        for i in range(numLimit):
            if contractLimit[i]:
                print('Set Contract Type Limit', i)
                if contractGeq[i]:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, contract[i]), basis)) >= \
                        contractLimit[i] * cp.sum(cp.multiply(x, basis)))
                else:
                    constraints.append(cp.sum(cp.multiply(cp.multiply(x, contract[i]), basis)) <= \
                        contractLimit[i] * cp.sum(cp.multiply(x, basis)))
    

    
    prob = cp.Problem(objective, constraints)
    return prob, x


def SolveModel(prob, timelimit=100):
    start_time = time.time()
    print("==============================================================")
    print('Model solving...')
    # solve model
    prob.solve(solver=cp.SCIP, verbose=True, scip_params={"limits/time": timelimit})

    print("==============================================================")
    print("status:", prob.status)
    print("==============================================================")
    print('Time Cost', time.time() - start_time)

    return prob

prob, x = BuildModel()

prob = SolveModel(prob, 100)











# import cvxpy as cp
# # Solving a problem with different solvers.
# x = cp.Variable(2)
# obj = cp.Minimize(x[0] + cp.norm(x, 1))
# constraints = [x >= 2]
# prob = cp.Problem(obj, constraints)

# # Solve with OSQP.
# prob.solve(solver=cp.OSQP)
# print("optimal value with OSQP:", prob.value)

# # Solve with ECOS.
# prob.solve(solver=cp.ECOS)
# print("optimal value with ECOS:", prob.value)

# # Solve with CVXOPT.
# prob.solve(solver=cp.CVXOPT)
# print("optimal value with CVXOPT:", prob.value)

# # Solve with SCS.
# prob.solve(solver=cp.SCS)
# print("optimal value with SCS:", prob.value)

# # Solve with GLPK.
# prob.solve(solver=cp.GLPK)
# print("optimal value with GLPK:", prob.value)

# # Solve with GLPK_MI.
# prob.solve(solver=cp.GLPK_MI)
# print("optimal value with GLPK_MI:", prob.value)

# # Solve with SCIP.
# prob.solve(solver=cp.SCIP)
# print("optimal value with SCIP:", prob.value)



# from pyscipopt import Model
# model = Model("Example")  # model name is optional

# x = model.addVar("x")
# y = model.addVar("y", vtype="INTEGER")
# model.setObjective(x + y)
# model.addCons(2*x - y*y >= 0)
# model.optimize()
# sol = model.getBestSol()
# print("x: {}".format(sol[x]))
# print("y: {}".format(sol[y]))



# import cvxpy as cp
# import numpy as  np
# import time
# n = 100000
# x = cp.Variable(shape=n, boolean=True)
# a = np.arange(n)
# b = np.ones(n)

# objective = cp.Maximize(cp.sum(cp.multiply(x, a)))
# # constraints
# constraints = [cp.sum(cp.multiply(x, b)) <= n/10]

# prob = cp.Problem(objective, constraints)

# start_time = time.time()
# print("==============================================================")
# print('Model solving...')
# # solve model
# prob.solve(solver=cp.SCIP, verbose=True, scip_params={"limits/time": 10})
# print("==============================================================")
# print("status:", prob.status)
# print("==============================================================")
# print('Time Cost', time.time() - start_time)
