#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 18:03
# @Author  : Pcc
# @Site    : 
# @File    : dps_database.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import MySQLdb as db

MAX_CHUNK = 1000


def connect_db():
    host = '127.0.0.1'
    port = 3306
    username = 'root'
    password = '123456'
    db_name = 'dps'

    try:
        mysql_cn = db.connect(host=host, port=port, user=username, passwd=password, db=db_name)
        return mysql_cn

    except db.MySQLError as e:
        print("Ops! Database Error: {}".format(e))


def db_select_with_param(sql, param, parse_date=None):
    mysql_cn = connect_db()
    df = pd.read_sql(sql=sql, con=mysql_cn, params=param, parse_dates=parse_date)
    mysql_cn.close()

    return df


def _pred_df_is_empty(pred_df):
    if pred_df.empty:
        print("Predict dataframe {} is empty! Please check the raw data!".format(pred_df.head()))
        return True
    else:
        return False


def db_select_latest_order_ids(limit_num):
    param = {"limit_num": limit_num}
    pre_stmt = "SELECT DISTINCT `productID` FROM " \
               "(SELECT `productID`, COUNT(`orderTime`) AS 'orderTimes' FROM `order` " \
               "GROUP BY `productID` ORDER BY 'orderTimes' LIMIT %(limit_num)s) AS temp_t;"

    df = db_select_with_param(sql=pre_stmt, param=param)

    id_tuples = tuple(list(df['productID']))

    return id_tuples


def db_select_order_ids_by_userId(user_id):
    param = {"user_id": user_id}
    pre_stmt = "SELECT `productID`, `userID` FROM `product` WHERE `productID` IN " \
               "(SELECT `productID` from `promotion` WHERE `effectTime` <= NOW() and `expireTime` >= NOW()) " \
               "AND `userID` = %(user_id)s;"

    df = db_select_with_param(sql=pre_stmt, param=param)

    id_tuples = tuple(list(df['productID']))

    return id_tuples


def db_select_pred_data_with_ids(id_tuples):
    # print("id_tuples: {}".format(id_tuples))
    if not id_tuples:
        print("Id_tuples: {} is empty, please check db query.".format(id_tuples))
        return None
    else:
        now = pd.Timestamp.now()
        num_rows = len(id_tuples)

        param= {"id_tuple": tuple(id_tuples)}

        pre_stmt1 = "SELECT `productID`, `highestPrice`, `lowestPrice`, `effectTime`, `expireTime` " \
                    "FROM `promotion` WHERE `productID` IN %(id_tuple)s " \
                    "AND `effectTime` <= NOW() AND `expireTime` >= NOW() " \
                    "ORDER BY `productID` DESC;"
        _time_df = db_select_with_param(pre_stmt1, param=param)
        _time_df['effectTime'] = pd.to_datetime(_time_df['effectTime'], format='%d%b%Y:%H:%M:%S.%f')
        _time_df['expireTime'] = pd.to_datetime(_time_df['expireTime'], format='%d%b%Y:%H:%M:%S.%f')

        pre_stmt2 = "SELECT `product`. `productID` AS 'productID', " \
                    "AVG(`product`.`initialPrice`) AS 'initialPrice', " \
                    "AVG(`order`.`inventoryRate`) AS 'inventoryRate', " \
                    "AVG(`order`.`productScore`) AS 'productScore', " \
                    "AVG(`order`.`vendorScore`) AS 'shopScore', " \
                    "AVG(`product`.`costPrice`) AS 'productCost'" \
                    "FROM `order` LEFT JOIN `product` ON `order`.`productID` = `product`.`productID` " \
                    "GROUP BY `product`. `productID` HAVING `product`.`productID` " \
                    "IN (SELECT `productID` FROM `promotion` WHERE `productID` " \
                    "IN %(id_tuple)s AND `expireTime` >= NOW() ORDER BY `productID` DESC) " \
                    "ORDER BY `product`. `productID` DESC;"
        df = db_select_with_param(pre_stmt2, param=param)

        print(_time_df)
        print('\n')
        print(df)
        if _pred_df_is_empty(df) or _pred_df_is_empty(_time_df):
            return None
        else:
            _time_df['affectTimeLeft'] = _time_df.apply(lambda x: ((now - x['effectTime']) / (x['expireTime'] - x['effectTime'])), axis=1)
            df['affectTimeLeft'] = _time_df['affectTimeLeft'].copy()
            df['highestPrice'] = _time_df['highestPrice'].copy()
            df['lowestPrice'] = _time_df['lowestPrice'].copy()
            df['targetPrice'] = pd.Series(data=np.zeros((num_rows, )), index=list(range(num_rows)))
            return df




