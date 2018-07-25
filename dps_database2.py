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
MAX_RATING = 5.0


def connect_db():
    host = '127.0.0.1'
    port = 3306
    username = 'root'
    # password = '123456'
    password = ''
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

    pre_stmt = "SELECT A.productID FROM `order` as A," \
               "(SELECT `productID`, max(`orderTime`) as max_time " \
               "FROM `order` GROUP BY `productID`) as B WHERE A.productID = B.productID " \
               "AND A.orderTime = B.max_time AND A.effectTimeLeft >= 0 ORDER BY B.max_time DESC LIMIT %(limit_num)s;"
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


def _normalize_score(dataframe, col_name, max_score):
    dataframe[col_name] = dataframe[col_name].apply(pd.to_numeric)
    dataframe[col_name] = dataframe.apply(
        lambda x: x[col_name] / max_score, axis=1)
    return dataframe


def db_select_pred_data_with_ids(id_tuples):
    print("id_tuples: {}".format(id_tuples))
    if not id_tuples:
        print("Id_tuples: {} is empty, please check db query.".format(id_tuples))
        return None
    else:
        num_rows = len(id_tuples)

        param= {"id_tuple": tuple(id_tuples)}

        pre_stmt1 = "SELECT `productID`, `highestPrice`, `lowestPrice` " \
                    "FROM `promotion` WHERE `productID` IN %(id_tuple)s " \
                    "ORDER BY `productID` DESC;"
        _time_df = db_select_with_param(pre_stmt1, param=param)

        pre_stmt2 = "SELECT `productID`, `initialPrice`, `costPrice` AS productCost FROM `product` " \
                    "WHERE `productID` IN %(id_tuple)s;"
        _product_df = db_select_with_param(pre_stmt2, param=param)

        pre_stmt3 = "SELECT A.`productID`, A.`inventoryRate`, A.`productScore`, A.`vendorScore` AS shopScore, " \
                    "A.`effectTimeLeft` AS affectTimeLeft FROM `order` as A," \
                    "(SELECT `productID`, max(`orderTime`) as max_time FROM `order` GROUP BY `productID`) as B " \
                    "WHERE A.productID = B.productID AND A.orderTime = B.max_time " \
                    "AND A.effectTimeLeft >=0 " \
                    "AND A.productID IN %(id_tuple)s" \
                    "ORDER BY B.max_time DESC;"

        # df = db_select_with_param(pre_stmt2, param=param)
        _order_df = db_select_with_param(pre_stmt3, param=param)

        df = pd.merge(_product_df, _order_df, how="left", on="productID")
        df = pd.merge(df, _time_df, how="left", on="productID")

        # print(_time_df)
        # print('\n')
        # print(df)
        if _pred_df_is_empty(_order_df) or _pred_df_is_empty(_product_df):
            return None
        else:
            df = _normalize_score(df, 'shopScore', MAX_RATING)
            df = _normalize_score(df, 'productScore', MAX_RATING)

            df['targetPrice'] = pd.Series(data=np.zeros((num_rows,)), index=list(range(num_rows)))
            return df


t = db_select_latest_order_ids(MAX_CHUNK)
df = db_select_pred_data_with_ids(t)
print(df)



