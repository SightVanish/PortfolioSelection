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
from scipy.sparse import csr_matrix
from pandasql import sqldf

INF = float('inf')
total_time = time.time()

if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)

parser = argparse.ArgumentParser(description="Flornes porfolio selection model")
parser.add_argument('--queryID', '-id', type=str, help='Query ID')
parser.add_argument('--threadLimit', '-t', type=int, default=4, help='Maximum number of threads')
parser.add_argument('--debug', action="store_true", help='Debug mode')
args = parser.parse_args().__dict__
threadLimit = args['threadLimit']
queryID = args['queryID']
debug = args['debug']
print('Input argparse',  args)

if queryID is None:
    print("No valid query id!")
    exit(1)

def ReportStatus(msg, flag, queryID, output=None):
    """
    Print message and update status in biz_model.biz_fir_query_parameter_definition.
    """
    sql = "update biz_model.biz_fir_third_parameter_definition set python_info_data='{0}', success_flag='{1}', update_time='{2}', python_result_json='{3}', version= version + 1 where query_id='{4}' and query_version = 2".format(msg, flag, datetime.datetime.now(), output, queryID)
    print("============================================================================================================================")
    print("Reporting issue:", msg)
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "biz_model_prod", user = "bizmodeluser", password = "$2kBBx@@!!")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    conn.close()

def ConnectDatabase(queryID):
    """
    Load parameters in JSON from biz_model.biz_fir_query_parameter_definition and load data from biz_model.biz_ads_fir_pkg_data.
    """
    try:
        print('Parameters reading...')
        sqlParameter = "select python_json from biz_model.biz_fir_query_parameter_definition where id='{0}'".format(queryID)
        conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "biz_model_prod", user = "bizmodeluser", password = "$2kBBx@@!!")
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
    try:
        print('Data loading...')
        sqlInput = \
        """
        select billing_status_fz as billing, unit_id_fz as unit_id, p1.product, fleet_year_fz as fleet_year, contract_cust_id as customer, p1.contract_num,
        contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age, ceu_fz as ceu, teu_fz as teu, rent as rent, rml_x_ceu_c as rml, cust_country
        from biz_model.biz_ads_fir_pkg_data p1
        inner join 
        (select contract_num, product
        from(
        select contract_num, product, count(*) num
        from biz_model.biz_ads_fir_pkg_data
        WHERE query_id='{1}'
        group by 1, 2
        ) p1 
        where num >= {0}) p2
        on p1.contract_num=p2.contract_num and p1.product=p2.product
        WHERE query_id='{1}'
        """.format(param["numContractProductLimit"], queryID)
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
    Output final package to biz_model.biz_fir_asset_package.
    """
    sqlOutput = "insert into biz_model.biz_fir_asset_package (unit_id, query_id, id, is_void, version, query_version) values %s"
    try:
        conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "biz_model_prod", user = "bizmodeluser", password = "$2kBBx@@!!")
        conn.autocommit = True
        cur = conn.cursor()
        print('Writing data...')
        values_list = []
        for i in range(len(result)):
            if result[i]:
                values_list.append((data['unit_id'][i], queryID, uuid.uuid1().hex, 0, 0, 2))
        psycopg2.extras.execute_values(cur, sqlOutput, values_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e) 
        ReportStatus("Writing data to GreenPlum Failed!", 'F', queryID)
        exit(1)

def DataProcessing(data):
    print("==============================================================")
    print('Data processing...')
    start_time = time.time()
    numData = data.shape[0]

    # Billing Status
    col = [i for i in range(numData)]
    row = [0 if data['billing'][i]=='ON' else 1 if data['billing'][i]=='OF' else 2 for i in range(numData)]
    statusOneHot = csr_matrix(([1 for _ in range(numData)], (row, col)), shape=(3, numData))

    # One hot all lessees
    lesseeIndex = {k: v for v, k in enumerate(data['customer'].value_counts().index)}
    row = [lesseeIndex[data['customer'][i]] for i in range(numData)]
    lesseeOneHot = csr_matrix(([1 for _ in range(numData)], (row, col)), shape=(len(data['customer'].value_counts()), numData))

    # One hot all contract number
    row = []
    contractIndex = {k: v for v, k in enumerate(data['contract_num'].value_counts().index)}
    row = [contractIndex[data['contract_num'][i]] for i in range(numData)]
    contractOneHot = csr_matrix(([1 for _ in range(numData)], (row, col)), shape=(len(data['contract_num'].value_counts()), numData))

    # One hot all contract type
    contractTypeIndex = {k: v for v, k in enumerate(data['contract'].value_counts().index)}
    row = [contractTypeIndex[data['contract'][i]] for i in range(numData)]
    contractTypeOneHot = csr_matrix(([1 for _ in range(numData)], (row, col)), shape=(len(data['contract'].value_counts()), numData))

    # One hot all product type
    productIndex = {k: v for v, k in enumerate(data['product'].value_counts().index)}
    row = [productIndex[data['product'][i]] for i in range(numData)]
    productOneHot = csr_matrix(([1 for _ in range(numData)], (row, col)), shape=(len(data['product'].value_counts()), numData))

    # Container Age
    row, col = [], [] # note: container age ranges may overlap
    for i in range(numData):
        for j in range(len(param['containersAge']['list'])):
            if param['containersAge']['list'][j]['containersAgeFrom'] <= data['fleet_year'][i] <= param['containersAge']['list'][j]['containersAgeTo']:
                row.append(j)
                col.append(i)
    containerAgeOneHot = csr_matrix(([1 for _ in range(len(row))], (row, col)), shape=(len(param['containersAge']['list']), numData))

    # Weighted Age
    row, col = [], []
    for i in range(numData):
        for j in range(len(param['weightedAge']['list'])):
            if param['weightedAge']['list'][j]['weightedAgeFrom'] <= data['weighted_age'][i] <= param['weightedAge']['list'][j]['weightedAgeTo']:
                row.append(j)
                col.append(i)
    weightedAgeOneHot = csr_matrix(([1 for _ in range(len(row))], (row, col)), shape=(len(param['weightedAge']['list']), numData))

    # RML
    row, col = [], []
    for i in range(numData):
        for j in range(len(param['rml']['list'])):
            if param['rml']['list'][j]['rmlFrom'] <= data['rml'][i] <= INF:
                row.append(j)
                col.append(i)
    rmlOneHot = csr_matrix(([1 for _ in range(len(row))], (row, col)), shape=(len(param['rml']['list']), numData))

    # Country
    row, col = [], []
    for i in range(numData):
        for j in range(len(param['country']['list'])):
            if data['cust_country'][i] in param['country']['list'][j]['country']:
                row.append(j)
                col.append(i)
    countryOneHot = csr_matrix(([1 for _ in range(len(row))], (row, col)), shape=(len(param['country']['list']), numData))

    print('Time cost:', time.time() - start_time)
    return statusOneHot, \
        lesseeIndex, lesseeOneHot, \
        contractIndex, contractOneHot, \
        contractTypeIndex, contractTypeOneHot, \
        productIndex, productOneHot, \
        containerAgeOneHot, weightedAgeOneHot, rmlOneHot, countryOneHot

def BuildModel(EnableWeightedAge=False, lookupTable=None):
    print("==============================================================")
    print('Building Model...')
    start_time = time.time()
    x = cp.Variable(shape=data.shape[0], boolean=True)
    x.value = np.ones(shape=data.shape[0])
    if EnableWeightedAge:
        y = cp.Variable(shape=lookupTable.shape[0], boolean=True)
        x = y @ lookupTable

    # objective
    objective = (x @ data['nbv']) if param['prefer']['nbvorCost'] else (x @ data['cost'])
    objective = cp.Maximize(objective) if param['prefer']['maxOrMin'] else cp.Minimize(objective)

    # constraints
    constraints = [cp.sum(x) >= 1]
    # NBV
    if param['totalNBVFrom']:
        print('Set NBV Lower Bound')
        constraints.append(x @ data['nbv'] >= param['totalNBVFrom'])
    if param['totalNBVTo']:
        print('Set NBV Upper Bound')
        constraints.append(x @ data['nbv'] <= param['totalNBVTo'])
    # Cost
    if param['totalCostFrom']:
        print('Set Cost Lower Bound')
        constraints.append(x @ data['cost'] >= param['totalCostFrom'])
    if param['totalCostTo']:
        print('Set Cost Upper Bound')
        constraints.append(x @ data['cost'] <= param['totalCostTo'])
    # Rent
    if param['totalRentFrom']:
        print('Set Rent Lower Bound')
        constraints.append(x @ data['rent'] >= param['totalRentFrom'])
    # Average Fleet Age
    if param['containersAge']['average']['averageContainersAge']:
        print('Set Average Container Age Limit')
        constraints.append(
            (1 if param['containersAge']['average']['symbol'] else -1) * (
            x @ data['fleet_year']
            - param['containersAge']['average']['averageContainersAge'] * cp.sum(x)
            ) >= 0)
    # Fleet Age
    for i in range(len(param['containersAge']['list'])):
        print(f'Set Container Age {i} Limit')
        constraints.append(
            (1 if param['containersAge']['list'][i]['symbol'] else -1) * (
            x @ (containerAgeOneHot[i].toarray().ravel() * data[param['containersAge']['basis']])
            - param['containersAge']['list'][i]['percent'] / 100 * (x @ data[param['containersAge']['basis']])
            ) >= 0)
    # Average Weighted Age
    if EnableWeightedAge:
        if param['weightedAge']['average']['averageWeighedAge']:
            print('Set Average Weighted Age Limit')
            constraints.append(
                (1 if param['weightedAge']['average']['symbol'] else -1) * (
                x @ (data['weighted_age'] * data['ceu'])
                - param['weightedAge']['average']['averageWeighedAge'] * (x @ data['ceu'])
                ) >= 0)
    # Weighted Age
    if EnableWeightedAge:
        for i in range(len(param['weightedAge']['list'])):
            print(f'Set Weighted Age {i} Limit')
            constraints.append(
                (1 if param['weightedAge']['list'][i]['symbol'] else -1) * (
                x @ (weightedAgeOneHot[i].toarray().ravel() * data['ceu'])
                - param['weightedAge']['list'][i]['percent'] / 100 * (x @ data['ceu'])
                ) >= 0)
    # RML
    for i in range(len(param['rml']['list'])):
        print(f'Set RML {i} Limit')
        constraints.append(
            (1 if param['rml']['list'][i]['symbol'] else -1) * (
            x @ (rmlOneHot[i].toarray().ravel() * data[param['rml']['basis']])
            - param['rml']['list'][i]['percent'] / 100 * (x @ data[param['rml']['basis']])
            ) >= 0)
    # Status
    for i in range(len(param['status']['list'])):
        print(f'Set Status {i} Limit')
        status = 0 if param['status']['list'][i]['statusType'] == 'ON' else 1 if param['status']['list'][i]['statusType'] == 'OF' else 2
        constraints.append(
            (1 if param['status']['list'][i]['symbol'] else -1) * (
            x @ (statusOneHot[status].toarray().ravel() * data[param['status']['basis']])
            - param['status']['list'][i]['percent'] / 100 * (x @ data[param['status']['basis']])
            ) >= 0)
    # Product Type
    for i in range(len(param['product']['list'])):
        print(f'Set Product Type {i} Limit')
        productListIndex = [productIndex.get(p) for p in param['product']['list'][i]['productType'] if productIndex.get(p) is not None]
        constraints.append(
            (1 if param['product']['list'][i]['symbol'] else -1) * (
            cp.sum(productOneHot[productListIndex].toarray() @ cp.multiply(x, data[param['product']['basis']]))
            - param['product']['list'][i]['percent'] / 100 * (x @ data[param['product']['basis']])
            ) >= 0)
    # Contract Type
    for i in range(len(param['contractType']['list'])):
        print(f'Set Contract Type {i} Limit')
        contractTypeListIndex = [contractTypeIndex.get(c) for c in param['contractType']['list'][i]['contractType'] if contractTypeIndex.get(c) is not None]
        constraints.append(
            (1 if param['contractType']['list'][i]['symbol'] else -1) * (
            cp.sum(contractTypeOneHot[contractTypeListIndex].toarray() @ cp.multiply(x, data[param['contractType']['basis']]))
            - param['contractType']['list'][i]['percent'] / 100 * (x @ data[param['contractType']['basis']])
            ) >= 0)
    # Country
    for i in range(len(param['country']['list'])):
        print(f'Set Country {i} Limit')
        constraints.append(
            (1 if param['country']['list'][i]['symbol'] else -1) * (
            x @ (countryOneHot[i].toarray().ravel() * data[param['country']['basis']])
            - param['country']['list'][i]['percent'] / 100 * (x @ data[param['country']['basis']])
            ) >= 0)
    # Certain Lessee
    for i in range(len(param['lessee']['list'])):
        print(f'Set Lessee {i} Limit')
        if param['lessee']['list'][i]['lessee'] not in lesseeIndex:
            print('\t{0} does not exist in dataset.'.format(param['lessee']['list'][i]['lessee']))
            continue
        constraints.append(
            (1 if param['lessee']['list'][i]['symbol'] else -1) * (
            x @ (lesseeOneHot[lesseeIndex[param['lessee']['list'][i]['lessee']]].toarray().ravel() * data[param['lessee']['basis']])
            - param['lessee']['list'][i]['percent'] / 100 * (x @ data[param['lessee']['basis']])
            ) >= 0)
    # Top Lessee
    for i in range(3):
        if param['lessee']['topLessee'][f'top{i+1}']['percent']:
            print(f'Set Top Lessee {i+1} Limit')
            constraints.append(
                cp.sum_largest(lesseeOneHot.toarray() @ cp.multiply(x, data[param['lessee']['basis']]), i + 1)
                - param['lessee']['topLessee'][f'top{i+1}']['percent'] / 100 * (x @ data[param['lessee']['basis']])
                <= 0)
    # Other Lessee -- only handle certain lessees
    if param['lessee']['others']['percent'] and param['lessee']['others']['lessee']:
        print('Set Other Lessees Limit')
        otherLesseeIndex = [lesseeIndex.get(l) for l in param['lessee']['others']['lessee'] if lesseeIndex.get(l) is not None]
        constraints.append(
            cp.sum_largest(lesseeOneHot[otherLesseeIndex].toarray() @ cp.multiply(x, data[param['lessee']['basis']]), 1)
            - param['lessee']['others']['percent'] / 100 * (x @ data[param['lessee']['basis']])
            <= 0)

    # Num Limit
    if not EnableWeightedAge:
        if param['numContractProductLimit']:
            print('Set Num Limit')
            contractProductType = [c.toarray().ravel()*p.toarray().ravel() for c in contractOneHot for p in productOneHot if sum(c.toarray().ravel()*p.toarray().ravel())>0]
            delta = cp.Variable(shape=len(contractProductType), boolean=True)
            for i in range(len(contractProductType)):
                constraints.append(x @ contractProductType[i] >= param['numContractProductLimit'] * delta[i])
                constraints.append(x @ contractProductType[i] <= 99999999 * delta[i])

    prob = cp.Problem(objective, constraints)
    print('Time Cost:', time.time() - start_time)
    
    return x, prob

def Validation(x, EnableWeightedAge=False):
    epsilon = 0.001
    passed = True
    print("==============================================================")
    print('Validating...')
    print(int(sum(x)), '/', len(x), 'containers are selected.')
    if sum(x) == 0:
        return False
    # objective
    objective = (x @ data['nbv']) if param['prefer']['nbvorCost'] else (x @ data['cost'])
    print('Objective: {0} = {1}'.format('nbv' if param['prefer']['nbvorCost'] else 'cost', objective))
    # NBV
    if param['totalNBVFrom']:
        p = x @ data['nbv'] >= param['totalNBVFrom'] - epsilon
        if not p: print('NBV Lower Bound Failed')
        passed = passed and p
    if param['totalNBVTo']:
        p = x @ data['nbv'] <= param['totalNBVTo'] + epsilon
        if not p: print('NBV Upper Bound Failed')
        passed = passed and p
    # Cost
    if param['totalCostFrom']:
        p = x @ data['cost'] >= param['totalCostFrom'] - epsilon
        if not p: print('Cost Lower Bound Failed')
        passed = passed and p
    if param['totalCostTo']:
        p = x @ data['cost'] <= param['totalCostTo'] + epsilon
        if not p: print('Cost Upper Bound Failed')
        passed = passed and p
    # Rent
    if param['totalRentFrom']:
        p = x @ data['rent'] >= param['totalRentFrom'] - epsilon
        if not p: print('Rent Lower Bound Failed')
        passed = passed and p
    # Average Fleet Age
    if param['containersAge']['average']['averageContainersAge']:
        p = (1 if param['containersAge']['average']['symbol'] else -1) * (
            x @ data['fleet_year']
            - param['containersAge']['average']['averageContainersAge'] * sum(x)
            ) >= - epsilon
        if not p: print('Average Fleet Age Failed')
        passed = passed and p
    # Fleet Age
    for i in range(len(param['containersAge']['list'])):
        p = (1 if param['containersAge']['list'][i]['symbol'] else -1) * (
            x @ (containerAgeOneHot[i].toarray().ravel() * data[param['containersAge']['basis']])
            - param['containersAge']['list'][i]['percent'] / 100 * (x @ data[param['containersAge']['basis']])
            ) >= - epsilon
        if not p: print(f'Fleet Age Limit {i} Failed')
        passed = passed and p
    # Average Weighted Age
    if EnableWeightedAge:
        if param['weightedAge']['average']['averageWeighedAge']:
            p = (1 if param['weightedAge']['average']['symbol'] else -1) * (
                x @ (data['weighted_age'] * data['ceu'])
                - param['weightedAge']['average']['averageWeighedAge'] * (x @ data['ceu'])
                ) >= - epsilon
            if not p: print('Average Weighted Age Failed')
            passed = passed and p
    # Weighted Age
    if EnableWeightedAge:
        for i in range(len(param['weightedAge']['list'])):
            p = (1 if param['weightedAge']['list'][i]['symbol'] else -1) * (
                x @ (weightedAgeOneHot[i].toarray().ravel() * data['ceu'])
                - param['weightedAge']['list'][i]['percent'] / 100 * (x @ data['ceu'])
                ) >= - epsilon
            if not p: print(f'Weighted Age Limit {i} Failed')
            if debug:
                print("selected containers, all containers, percentage 100%")
                print(x @ (weightedAgeOneHot[i].toarray().ravel() * data['ceu']), (x @ data['ceu']), param['weightedAge']['list'][i]['percent'])
            passed = passed and p
    # RML
    for i in range(len(param['rml']['list'])):
        p = (1 if param['rml']['list'][i]['symbol'] else -1) * (
            x @ (rmlOneHot[i].toarray().ravel() * data[param['rml']['basis']])
            - param['rml']['list'][i]['percent'] / 100 * (x @ data[param['rml']['basis']])
            ) >= - epsilon
        if not p: print(f'RML Limit {i} Failed')
        passed = passed and p
    # Status
    for i in range(len(param['status']['list'])):
        status = 0 if param['status']['list'][i]['statusType'] == 'ON' else 1 if param['status']['list'][i]['statusType'] == 'OF' else 2
        p = (1 if param['status']['list'][i]['symbol'] else -1) * (
            x @ (statusOneHot[status].toarray().ravel() * data[param['status']['basis']])
            - param['status']['list'][i]['percent'] / 100 * (x @ data[param['status']['basis']])
            ) >= - epsilon
        if not p: print(f'Status {i} Limit Failed')
        passed = passed and p
    # Product Type
    for i in range(len(param['product']['list'])):
        productListIndex = [productIndex.get(p) for p in param['product']['list'][i]['productType'] if productIndex.get(p) is not None]
        p = (1 if param['product']['list'][i]['symbol'] else -1) * (
            sum(productOneHot[productListIndex].toarray() @ (x * data[param['product']['basis']]))
            - param['product']['list'][i]['percent'] / 100 * (x @ data[param['product']['basis']])
            ) >= - epsilon
        if not p: print(f'Product Type {i} Limit Failed')
        passed = passed and p
    # Contract Type
    for i in range(len(param['contractType']['list'])):
        contractTypeListIndex = [contractTypeIndex.get(c) for c in param['contractType']['list'][i]['contractType'] if contractTypeIndex.get(c) is not None]
        p = (1 if param['contractType']['list'][i]['symbol'] else -1) * (
            sum(contractTypeOneHot[contractTypeListIndex].toarray() @ (x * data[param['contractType']['basis']]))
            - param['contractType']['list'][i]['percent'] / 100 * (x @ data[param['contractType']['basis']])
            ) >= - epsilon
        if not p: print(f'Contract Type {i} Limit Failed')
        passed = passed and p
    # Country
    for i in range(len(param['country']['list'])):
        p = (1 if param['country']['list'][i]['symbol'] else -1) * (
            x @ (countryOneHot[i].toarray().ravel() * data[param['country']['basis']])
            - param['country']['list'][i]['percent'] / 100 * (x @ data[param['country']['basis']])
            ) >= - epsilon
        if not p: print(f'Country {i} Limit Failed')
        passed = passed and p
    # Certain Lessee
    for i in range(len(param['lessee']['list'])):
        if param['lessee']['list'][i]['lessee'] not in lesseeIndex:
            print('\t{0} does not exist in dataset.'.format(param['lessee']['list'][i]['lessee']))
            continue
        p = (1 if param['lessee']['list'][i]['symbol'] else -1) * (
            x @ (lesseeOneHot[lesseeIndex[param['lessee']['list'][i]['lessee']]].toarray().ravel() * data[param['lessee']['basis']])
            - param['lessee']['list'][i]['percent'] / 100 * (x @ data[param['lessee']['basis']])
            ) >= - epsilon
        if not p: print(f'Lessee {i} Limit Failed')
        passed = passed and p
    # Top Lessee
    for i in range(3):
        if param['lessee']['topLessee'][f'top{i+1}']['percent']:
            p = sum(heapq.nlargest(i + 1, lesseeOneHot.toarray() @ (x * data[param['lessee']['basis']]))) \
                - param['lessee']['topLessee'][f'top{i+1}']['percent'] / 100 * (x @ data[param['lessee']['basis']]) \
                <= epsilon
            if not p: print(f'Top {i+1} Lessee Failed')
            passed = passed and p
    # Other Lessee -- only handle certain lessees
    if param['lessee']['others']['percent'] and param['lessee']['others']['lessee']:
        otherLesseeIndex = [lesseeIndex.get(l) for l in param['lessee']['others']['lessee'] if lesseeIndex.get(l) is not None]
        p = max(lesseeOneHot[otherLesseeIndex].toarray() @ (x * data[param['lessee']['basis']])) \
            - param['lessee']['others']['percent'] / 100 * (x @ data[param['lessee']['basis']]) \
            <= epsilon
        if not p: print('Other Lessees Failed')
        passed = passed and p
    # Num Limit
    if not EnableWeightedAge:
        if param['numContractProductLimit']:
            contractProductType = [c.toarray().ravel()*p.toarray().ravel() for c in contractOneHot for p in productOneHot]
            p = min([c @ x for c in contractProductType if c @ x > 0]) >= param['numContractProductLimit'] - 1
            if not p: print('Num Limit Failed')
            passed = passed and p
    return passed

param, data = ConnectDatabase(queryID)
data = data.fillna("None")

try:
    model_time = time.time()
    statusOneHot, \
    lesseeIndex, lesseeOneHot, \
    contractIndex, contractOneHot, \
    contractTypeIndex, contractTypeOneHot, \
    productIndex, productOneHot, \
    containerAgeOneHot, weightedAgeOneHot, rmlOneHot, countryOneHot \
    = DataProcessing(data)
    x, prob = BuildModel()
    print("==============================================================")
    print('Model solving...')
    prob.solve(solver=cp.CBC, verbose=debug, maximumSeconds=max(param['timeLimit'], 100), numberThreads=threadLimit)
    print("Status:", prob.status)
    print('Time Cost', time.time() - model_time)
except Exception as e:
    print(e)
    ReportStatus('Model Failed! Please Contact Developing Team!', 'F', queryID)
    exit(1)

if prob.status == 'infeasible':
    ReportStatus('Problem Proven Infeasible! Please Modify Constaints.', 'I', queryID)
    exit(0)
try:
    x = np.where(abs(x.value-1) < 1e-3, 1, 0) # x == 1
    passed = Validation(x)
except Exception as e:
    print(e)
    ReportStatus('Validation Failed! Please Contact Developing Team!', 'F', queryID)
    exit(1)

if not passed:
    OutputPackage(data, x, queryID)
    ReportStatus('Constraints Cannot Be fulfilled! Please Modify Constaints Or Increase Running Timelimit.', 'N', queryID)
    exit(0)

# Passed
if not param['weightedAge']['list']:
    # if no limit on weighted age
    OutputPackage(data, x, queryID)
    ReportStatus('Algorithm Succeeded!', 'O', queryID)
    exit(0)

print("==============================================================")
print('Handling Weighted Age Limit Now...')
# Handle Weighted Age
try:
    # data processing
    data = data.iloc[list(np.nonzero(x)[0])].reset_index()

    print("Executing Pandas SQL...")
    pysqldf = lambda q: sqldf(q, globals())
    q = \
    """
    select billing, unit_id, p1.product, fleet_year, customer, p1.contract_num,
    contract, cost, nbv, weighted_age, ceu, teu, rent, rml, cust_country
    from data p1
    inner join 
    (select contract_num, product
    from (select contract_num, product, count(*) num from data group by 1, 2) p1 
    where num >= {0}) p2
    on p1.contract_num=p2.contract_num and p1.product=p2.product
    """.format(param["numContractProductLimit"])
    data = pysqldf(q)
    print('Input data shape:', data.shape)

    # compute lookup table
    contract_product = {}
    j = 0
    for i in range(data.shape[0]):
        if (data['contract_num'][i], data['product'][i]) not in contract_product:
            contract_product[(data['contract_num'][i], data['product'][i])] = j
            j += 1
    row, col = [], []
    for i in range(data.shape[0]):
        row.append(contract_product[(data['contract_num'][i], data['product'][i])])
        col.append(i)
    value = [1 for _ in range(data.shape[0])]
    lookupTable = csr_matrix((value, (row, col)), shape=(len(contract_product), data.shape[0]))

    model_time = time.time()
    statusOneHot, \
    lesseeIndex, lesseeOneHot, \
    contractIndex, contractOneHot, \
    contractTypeIndex, contractTypeOneHot, \
    productIndex, productOneHot, \
    containerAgeOneHot, weightedAgeOneHot, rmlOneHot, countryOneHot \
    = DataProcessing(data)
    x, prob = BuildModel(EnableWeightedAge=True, lookupTable=lookupTable)
    print("==============================================================")
    print('Model solving...')
    prob.solve(solver=cp.CBC, verbose=debug, maximumSeconds=max(param['timeLimit'], 100), numberThreads=threadLimit)
    print("Status:", prob.status)
    print('Time Cost', time.time() - model_time)

except Exception as e:
    print(e)
    ReportStatus('Model Failed! Please Contact Developing Team!', 'F', queryID)
    exit(1)

if prob.status == 'infeasible':
    ReportStatus('Constraints on Weighted Age Cannot Be fulfilled! Please Modify Constaints.', 'WF', queryID)
    OutputPackage(data, [1 for _ in range(data.shape[0])], queryID) # output all
    exit(0)

try:
    x = np.where(abs(x.value-1) < 1e-3, 1, 0) # x == 1
    passed = Validation(x, EnableWeightedAge=True)
except Exception as e:
    print(e)
    ReportStatus('Validation Failed! Please Contact Developing Team!', 'F', queryID)
    exit(1)

if not passed:
    OutputPackage(data, [1 for _ in range(data.shape[0])], queryID) # output all
    ReportStatus('Constraints on Weighted Age Cannot Be fulfilled! Please Modify Constaints.', 'WF', queryID)
    exit(0)
else:
    OutputPackage(data, x, queryID)
    ReportStatus('Algorithm Succeeded!', 'O', queryID)
    exit(0)
