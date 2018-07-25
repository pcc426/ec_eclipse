#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 15:05
# @Author  : Pcc
# @Site    : 
# @File    : test_demo.py
# @Software: PyCharm


import pandas as pd

df = pd.read_csv("./data/gen_data_2018-07-25_v1.csv")
# tmp = df.sum(axis=0)
tmp = df["Total_Pay"].count()

# for index, row in df.iterrows():
#     # print(row['c1'].tolist())
#     print(list(row['c1'].tolist()))

# t = tuple(list(df['c1']))
print(tmp)