#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 17:49
# @Author  : Pcc
# @Site    : 
# @File    : index.py
# @Software: PyCharm

import json

test_dict = {
    "tarPrice": 11.,
    "message": "great!"
}
j_test = json.dumps(test_dict)
# print("Hello! Just test Python!")
print(j_test)
