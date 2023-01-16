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

INF = float('inf')
total_time = time.time()

if sys.version_info[0:2] != (3, 6):
    warnings.warn('Please use Python3.6', UserWarning)

queryID = "u4xf1eivepKM9EQHrefgK0"
numLimit = 5
threadLimit = 4
if queryID is None:
    print("No valid query id!")
    exit(1)

def ReportStatus(msg, flag, queryID, output=None):
    """
    Print message and update status in biz_model.biz_fir_query_parameter_definition.
    """
    sql = "update fll_t_dw.biz_fir_query_parameter_definition set python_info_data='{0}', success_flag='{1}', update_time='{2}', python_result_json='{3}' where id='{4}'".format(msg, flag, datetime.datetime.now(), output, queryID)
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
    Load parameters in JSON from biz_model.biz_fir_query_parameter_definition and load data from biz_model.biz_ads_fir_pkg_data.
    """
    print('Parameters reading...')
    sqlParameter = "select python_json from fll_t_dw.biz_fir_query_parameter_definition where id='{0}'".format(queryID)
    conn = psycopg2.connect(host = "10.18.35.245", port = "5432", dbname = "iflorensgp", user = "fluser", password = "13$vHU7e")        
    paramInput = pd.read_sql(sqlParameter, conn)
    if paramInput.shape[0] == 0:
        raise Exception("No Valid Query Request is Found!")
    elif paramInput.shape[0] > 1:
        raise Exception("More than One Valid Query Requests are Found!")
    param = json.loads(paramInput['python_json'][0])
    # print(param)

    print('Data loading...')
    # sqlInput = """
    #     select billing_status_fz as billing, unit_id_fz as unit_id, product, fleet_year_fz as fleet_year, contract_cust_id as customer, contract_num as contract_num, \
    #     contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age, ceu_fz as ceu, teu_fz as teu, rent as rent, rml_x_ceu as rml
    #     from fll_t_dw.biz_ads_fir_pkg_data WHERE query_id='{0}'
    # """.format(queryID) 

    param["numContractProductLimit"] = 150
    sqlInput = \
    """
    select billing_status_fz as billing, unit_id_fz as unit_id, p1.product, fleet_year_fz as fleet_year, contract_cust_id as customer, p1.contract_num,
    contract_lease_type as contract, cost, nbv, age_x_ceu as weighted_age, ceu_fz as ceu, teu_fz as teu, rent as rent, rml_x_ceu as rml
    from fll_t_dw.biz_ads_fir_pkg_data p1
    inner join 
    (select contract_num, product
    from(
    select contract_num, product, count(*) num
    from fll_t_dw.biz_ads_fir_pkg_data
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


    return param, data


param, data = ConnectDatabase(queryID)

# print('Data reading...')
# data = pd.read_csv('./local_data.csv')
print('Parameter loading...')
with open("./parameterDemoTest.json") as f:
    param = json.load(f)
# queryID = "local_test_id"
print("==============================================================")
print(param)
print(data.shape)
