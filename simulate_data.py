#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Pcc on 2018/6/13

import numpy as np
import pandas as pd
import datetime


# generate data for a particular date
def gen_data_frame_for_date(date, total_pay, average_pay, shop_score_mean=0.6, gross_margin=0.2):
    StdInitialPrice = 5.
    StdInitialShopScore = 0.5

    # initial random seed
    np.random.seed(np.random.randint(1, 30))
    # np.random.seed(30)

    _initial_price = StdInitialPrice * np.random.randn(total_pay) + average_pay  # initialize initial_price by normal distribution
    _initial_price = _initial_price.astype(int)  # round up initial_price to int
    _shop_score = StdInitialShopScore * np.random.randn(total_pay) + shop_score_mean
    rand_series = pd.Series(np.random.random(total_pay), index=list(range(total_pay)))

    df = pd.DataFrame({
        'Trade_Date': pd.Timestamp(date),
        'Initial_Price': pd.Series(_initial_price, index=list(range(total_pay))),
        # 'Inventory': rand_series,
        'Inventory': pd.Series(np.random.uniform(0, 1, total_pay), index=list(range(total_pay))),
        'Total_Pay': pd.Series(total_pay, index=list(range(total_pay))),
        'Product_Score': rand_series,
        'Shop_Score': pd.Series(_shop_score, index=list(range(total_pay))),
        # 'Shop_Score': pd.Series(np.random.uniform(0, 1, total_pay), index=list(range(total_pay))),
        'Active_Time_Left': pd.Series(np.random.random(total_pay), index=list(range(total_pay))),

    })

    # df['Product_Cost'] = pd.Series(df['Initial_Price'] * 0.8, index=list(range(RowNum)))
    df['Product_Cost'] = df.apply(lambda row: row['Initial_Price'] * (1 - gross_margin), axis=1)
    df['Target_Price'] = pd.Series(0., index=list(range(total_pay)))
    condition1 = df.loc[:, 'Shop_Score'] < 0
    condition2 = df.loc[:, 'Shop_Score'] > 1
    df.Shop_Score[condition1] = 0.  # assign negative shop_score value to 0
    df.Shop_Score[condition2] = 1.  # assign shop_score over 1 to 1

    return df


# print(df1)

# generate data for all dates in original query results
def gen_all_data(pay_count_csv, average_pay):
    df_all = pd.DataFrame(columns=['Trade_Date',
                                   'Initial_Price',
                                   'Inventory',
                                   'Total_Pay',
                                   'Product_Score',
                                   'Shop_Score',
                                   'Active_Time_Left',
                                   'Product_Cost',
                                   'Target_Price'])

    df_pay_count = pd.read_csv(pay_count_csv)  # import pay_count
    for index, row in df_pay_count.iterrows():
        # print(row['Date'], row['Pay_count'])
        df_date = gen_data_frame_for_date(row['Date'], row['Pay_count'], average_pay=average_pay)
        df_all = df_all.append(df_date, ignore_index=True)

    return df_all


df = gen_all_data('./data/query_result_payCount.csv', average_pay=17.0)
df.to_csv('./data/gen_data_' + str(datetime.date.today()) + '.csv')

