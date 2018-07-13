#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 17:42
# @Author  : Pcc
# @Site    :
# @File    : dps_data.py
# @Software: PyCharm

import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

TRAIN_PATH = './pcc_data/csv_result-gen_data_2018-06-19_target price_train.csv'
TEST_PATH = './pcc_data/csv_result-gen_data_2018-06-19_target price_test.csv'
ALL_PATH = './pcc_data/csv_result-gen_data_2018-06-19_target price_clean.csv'


CSV_COLUMN_NAMES = ['Initial_Price', 'Inventory_Rate',
                    'Product_Score', 'Shop_Score', 'Active_Time_Left',
                    'Product_Cost', 'Target_Price']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name='Target_Price'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    # train_path, test_path = maybe_download()

    all = pd.read_csv(ALL_PATH, names=CSV_COLUMN_NAMES, header=0)
    train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0)

    # Normalized data
    train, test = normalize_data(all, train, test)

    train_x, train_y = train, train.pop(y_name)
    print("1st train_y: {}".format(train_y[0]))

    test_x, test_y = test, test.pop(y_name)
    print("1st test_y: {}".format(test_y[0]))

    return (train_x, train_y), (test_x, test_y)


def normalize_data(all_data_set, train_set, test_set):
    # Normalize the data
    mean = all_data_set.mean(axis=0)
    std = all_data_set.std(axis=0)
    normalized_train = (train_set - mean) / std
    normalized_test = (test_set - mean) / std
    return normalized_train, normalized_test


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def test_input_fn(test_set, batch_size):
    return (test_set.shuffle(1000).batch(batch_size)
            .make_one_shot_iterator().get_next())


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

# train = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0)
# print(train)