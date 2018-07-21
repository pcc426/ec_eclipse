#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Pcc on 2018/6/14

import numpy as np
import pandas as pd
import MySQLdb as db
import argparse
import sys


import tf_predict as predict
import dps_database as dps_db

DEFAULT_USER_ID = 1


def _get_price_id_list(pred_df):
    price_id_list = []
    # if not dps_db._pred_df_is_empty(pred_df):
    df = pred_df
    price_id_list = list(zip(df['targetPrice'], df['productID']))
    # print(price_id_list)
    return price_id_list


def db_update_current_price_by_id(price_id_list):
    mysql_cn = dps_db.connect_db()
    query = "UPDATE `product` SET `currentPrice` = %s WHERE `productID` = %s;"

    try:
        cur = mysql_cn.cursor()
        cur.executemany(query, args=price_id_list)
        mysql_cn.commit()
        return 1
    except mysql_cn.Error as e:
        print("Ops! Database Error: {}".format(e))
        return 0
    finally:
        mysql_cn.close()


# Not working for tf_predict doesn't support multi-outputs yet!
def _convert_df_to_list_dict(dataframe):
    predict_x = {
        'Initial_Price': dataframe['initialPrice'].tolist(),
        'Inventory_Rate': dataframe['inventoryRate'].tolist(),
        'Product_Score': dataframe['productScore'].tolist(),
        'Shop_Score': dataframe['shopScore'].tolist(),
        'Active_Time_Left': dataframe['affectTimeLeft'].tolist(),
        'Product_Cost': dataframe['productCost'].tolist(),
    }
    return predict_x, dataframe


def _convert_df_row_to_list_dict(row):
    predict_x = {
        'Initial_Price': row['initialPrice'].tolist(),
        'Inventory_Rate': row['inventoryRate'].tolist(),
        'Product_Score': row['productScore'].tolist(),
        'Shop_Score': row['shopScore'].tolist(),
        'Active_Time_Left': row['affectTimeLeft'].tolist(),
        'Product_Cost': row['productCost'].tolist(),
    }
    return predict_x


def predict_current_price(pred_df, product_id):
    exp_file_path = predict.EXPORT_MODEL_FILE + str(product_id) + ".txt"
    saved_model_dir = predict.extract_model_dir(exp_file_path)

    # if not dps_db._pred_df_is_empty(pred_df):
    for index, row in pred_df.iterrows():
                predict_x = {
                    'Initial_Price': [row['initialPrice']],
                    'Inventory_Rate': [row['inventoryRate']],
                    'Product_Score': [row['productScore']],
                    'Shop_Score': [row['shopScore']],
                    'Active_Time_Left': [row['affectTimeLeft']],
                    'Product_Cost': [row['productCost']],
                }
                prediction_list = predict.predict_from_saved_model(saved_model_dir, predict_x)
                prediction_list = predict.prune_predictions(prediction_list)
                pred_price = prediction_list[0][0]

                max_price = row['highestPrice']
                min_price = row['lowestPrice']
                df.at[index, 'targetPrice'] = _prune_price(pred_price, max_price, min_price)

    return pred_df


def _prune_price(pred_price, max_price, min_price):
    if pred_price > max_price:
        return max_price
    elif pred_price < min_price:
        return min_price
    else:
        return round(pred_price, 0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--product_id', default="0000", type=str, help='product id')
    args = parser.parse_args(sys.argv[1:])

    # Only update prices for latest trading products!
    ids_tuple = dps_db.db_select_latest_order_ids(limit_num=dps_db.MAX_CHUNK)
    # ids_tuple = dps_db.db_select_order_ids_by_userId(DEFAULT_USER_ID)
    # Select predict_x features according to retrieved ids
    df = dps_db.db_select_pred_data_with_ids(id_tuples=ids_tuple)
    # Predict new current prices
    # df = predict_current_price(pred_df=df, product_id=str(args.product_id))
    df = predict_current_price(pred_df=df, product_id=str(args.product_id))
    print(df)
    pl = _get_price_id_list(pred_df=df)
    # Update currentPrice in database
    return_num = db_update_current_price_by_id(pl)

    if return_num == 0:
        print("Ops! Something goes wrong! Update price fails!")
    else:
        print("Yeah! Current prices have been updated!")





