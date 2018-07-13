#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 17:42
# @Author  : Pcc
# @Site    :
# @File    : tf_build_model.py
# @Software: PyCharm


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import itertools

import dps_data


# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# parser.add_argument('--train_steps', default=1500, type=int,
#                     help='number of training steps')

DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAIN_STEPS = 1500
EXPORT_MODEL_FILE = './export_files/export_file_'
EXPORT_MODEL_DIR = './pcc_models/exported_'


def train_and_save_model(batch_size=DEFAULT_BATCH_SIZE, train_steps=DEFAULT_TRAIN_STEPS,
                         product_id='0000', exp_model_file=EXPORT_MODEL_FILE,
                         exp_model_dir=EXPORT_MODEL_DIR):
    # args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = dps_data.load_data()
    print('train_x shape:' + str(train_x.shape))
    print('test_x shape:' + str(test_x.shape))

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Checkpoint config
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=1,  # Retain the 1 most recent checkpoints.
    )

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    model = tf.estimator.DNNRegressor(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # Customize optimizer with learning_rate and regulation param.
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        ),
        # model_dir='pcc_models/demo1',
        # config=my_checkpointing_config
    )

    # Train the Model.
    model.train(
        input_fn=lambda:dps_data.train_input_fn(train_x, train_y, batch_size=batch_size),
        steps=train_steps)

    # Evaluate the model.
    eval_result = model.evaluate(
        input_fn=lambda:dps_data.eval_input_fn(test_x, test_y, batch_size=batch_size))

    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\n" + 80 * "*")
    # print("\nRMS error for the test set: ${:.0f}"
    #       .format(PRICE_NORM_FACTOR * average_loss ** 0.5))

    print("average_loss: {}".format(average_loss))

    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = [20., 23.5, 26.5, 30.]
    # predict_x = {
    #     'Initial_Price': [16., 22., 26., 25.],
    #     'Inventory_Rate': [0.156091, 0.156091, 0.156091, 0.845345],
    #     'Product_Score': [0.477172, 0.477172, 0.477172, 0.796761],
    #     'Shop_Score': [0.628411, 0.82612, 0.75636, 1.],
    #     'Active_Time_Left': [0.595063, 0.238082, 0.338095, 0.36078],
    #     'Product_Cost': [12.8, 17.6, 20.8, 20],
    # }

    predict_x = {
        'Initial_Price': [16.],
        'Inventory_Rate': [0.156091],
        'Product_Score': [0.477172],
        'Shop_Score': [0.628411],
        'Active_Time_Left': [0.595063],
        'Product_Cost': [12.8],
    }

    predictions = model.predict(
        input_fn=lambda:dps_data.eval_input_fn(predict_x, labels=None, batch_size=batch_size))

    for pred, expec in zip(predictions, expected):
        print("\n prediction: {}, expected: {}".format(pred['predictions'], expec))

    # There's still some problems in feature_spec, maybe need to rewrite output tensor in Predictor
    # feature_spec = {'Initial_Price': tf.FixedLenSequenceFeature(shape=[4], dtype=tf.float32, allow_missing=True),
    #                 'Inventory_Rate': tf.FixedLenSequenceFeature(shape=[4],dtype=tf.float32, allow_missing=True),
    #                 'Product_Score': tf.FixedLenSequenceFeature(shape=[4], dtype=tf.float32, allow_missing=True),
    #                 'Shop_Score': tf.FixedLenSequenceFeature(shape=[4], dtype=tf.float32, allow_missing=True),
    #                 'Active_Time_Left': tf.FixedLenSequenceFeature(shape=[4], dtype=tf.float32, allow_missing=True),
    #                 'Product_Cost': tf.FixedLenSequenceFeature(shape=[4], dtype=tf.float32, allow_missing=True),
    #                 }

    # There's still some problems in feature_spec, maybe need to rewrite output tensor in Predictor
    feature_spec = {'Initial_Price': tf.FixedLenFeature(shape=[], dtype=tf.float32),
                    'Inventory_Rate': tf.FixedLenFeature(shape=[], dtype=tf.float32),
                    'Product_Score': tf.FixedLenFeature(shape=[], dtype=tf.float32),
                    'Shop_Score': tf.FixedLenFeature(shape=[], dtype=tf.float32),
                    'Active_Time_Left': tf.FixedLenFeature(shape=[], dtype=tf.float32),
                    'Product_Cost': tf.FixedLenFeature(shape=[], dtype=tf.float32),
                    }

    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=None,
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        # print(features)

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    # default serving_input_receiver
    saved_model_dir = model.export_savedmodel(exp_model_dir + str(product_id), serving_input_receiver_fn,
                                              strip_default_attrs=True)

    # Write export_dir to a txt file
    with open(exp_model_file + str(product_id) + '.txt', 'w+') as f:
        f.write(str(saved_model_dir))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--product_id', default="0000", type=str, help='product id')

    args = parser.parse_args(argv[1:])
    train_and_save_model(DEFAULT_BATCH_SIZE, DEFAULT_TRAIN_STEPS, product_id=str(args.product_id))


# main执行需调整，不用输出log
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
