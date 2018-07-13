#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 15:05
# @Author  : Pcc
# @Site    : 
# @File    : test_demo.py
# @Software: PyCharm


import pandas as pd
inp = [{'c1':10, 'c2':100}, {'c1':11,'c2':110}, {'c1':12,'c2':120}]
df = pd.DataFrame(inp)
print(df)

# for index, row in df.iterrows():
#     # print(row['c1'].tolist())
#     print(list(row['c1'].tolist()))

t = tuple(list(df['c1']))
print(t)