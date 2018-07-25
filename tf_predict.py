#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 17:42
# @Author  : Pcc
# @Site    :
# @File    : tf_predict.py
# @Software: PyCharm

import tensorflow as tf

import tf_build_model
import os

EXPORT_MODEL_FILE = tf_build_model.EXPORT_MODEL_FILE
MAX_PRICE = 1000.
MIN_PRICE = 0.
pat_path = os.path.abspath("/Users/pcc/ec_eclipse")


def predict_from_saved_model(saved_model_dir, predict_x):

    # saved_model_dir = saved_model_dir + str(product_id)

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        predictor = tf.contrib.predictor.from_saved_model(saved_model_dir)
        model_input = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "Initial_Price": tf.train.Feature(float_list=tf.train.FloatList(value=predict_x['Initial_Price'])),
                    "Inventory_Rate": tf.train.Feature(float_list=tf.train.FloatList(value=predict_x['Inventory_Rate'])),
                    "Product_Score": tf.train.Feature(float_list=tf.train.FloatList(value=predict_x['Product_Score'])),
                    "Shop_Score": tf.train.Feature(float_list=tf.train.FloatList(value=predict_x['Shop_Score'])),
                    "Active_Time_Left": tf.train.Feature(float_list=tf.train.FloatList(value=predict_x['Active_Time_Left'])),
                    "Product_Cost": tf.train.Feature(float_list=tf.train.FloatList(value=predict_x['Product_Cost'])),

                })
        )
        model_input = model_input.SerializeToString()
        # print(model_input)
        output_dict = predictor({"inputs": [model_input]})
        # print('Prediction: {}'.format(output_dict['outputs'][0][0]))
        predictions = output_dict['outputs'].tolist()
        print(predictions[0])

    return predictions


def extract_model_dir(file_path):
    with open(file_path, 'r+') as f:
        file_name = f.readline()
        file_name = pat_path + file_name[3:-1]

    return file_name


#  Limit prediction price in a range
def prune_predictions(prediction_list, max_price=MAX_PRICE, min_price=MIN_PRICE):
    for price in prediction_list:
        if price[0] > max_price:
            price[0] = max_price
        elif price[0] < min_price:
            price[0] = min_price

    return prediction_list


# predict_x = {
#         'Initial_Price': [16., 22., 26., 25.],
#         'Inventory_Rate': [0.156091, 0.156091, 0.156091, 0.845345],
#         'Product_Score': [0.477172, 0.477172, 0.477172, 0.796761],
#         'Shop_Score': [0.628411, 0.82612, 0.75636, 1.],
#         'Active_Time_Left': [0.595063, 0.238082, 0.338095, 0.36078],
#         'Product_Cost': [12.8, 17.6, 20.8, 20],
#     }

predict_x = {
        'Initial_Price': [16.],
        'Inventory_Rate': [0.156091],
        'Product_Score': [0.477172],
        'Shop_Score': [0.628411],
        'Active_Time_Left': [0.595063],
        'Product_Cost': [12.8],
    }


if __name__ == '__main__':
    saved_model_dir = extract_model_dir(EXPORT_MODEL_FILE + "0000.txt")
    p = predict_from_saved_model(saved_model_dir, predict_x)
    p = prune_predictions(p)
    print("Prediction: {}".format(p))
    # To PHP, 直接print即可; to c#, 可能需要借助Json和sys.stdout








