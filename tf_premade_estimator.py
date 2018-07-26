#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import itertools

import dps_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_steps', default=1500, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

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
        # model_dir='expected_models/demo1',
        # config=my_checkpointing_config
    )

    # Train the Model.
    model.train(
        input_fn=lambda:dps_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = model.evaluate(
        input_fn=lambda:dps_data.eval_input_fn(test_x, test_y, args.batch_size))

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
    predict_x = {
        'Initial_Price': [16., 22., 26., 25.],
        'Inventory_Rate': [0.156091, 0.156091, 0.156091, 0.845345],
        'Product_Score': [0.477172, 0.477172, 0.477172, 0.796761],
        'Shop_Score': [0.628411, 0.82612, 0.75636, 1.],
        'Active_Time_Left': [0.595063, 0.238082, 0.338095, 0.36078],
        'Product_Cost': [12.8, 17.6, 20.8, 20],
    }

    predictions = model.predict(
        input_fn=lambda:dps_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    # print("\n Prediction: {}, expected: {}".format(predictions.flatten, expected))
    # predictions = list(itertools.islice(predictions, len(expected)))
    # print("Predictions: {}".format(str(predictions)))
    for pred, expec in zip(predictions, expected):
        print("\n prediction: {}, expected: {}".format(pred['predictions'], expec))


# 需要完成模型的导出

    feature_spec = {'Initial_Price': tf.FixedLenFeature([2], dtype=tf.float32),
                    'Inventory_Rate': tf.FixedLenFeature([2], dtype=tf.float32),
                    'Product_Score': tf.FixedLenFeature([2], dtype=tf.float32),
                    'Shop_Score': tf.FixedLenFeature([2], dtype=tf.float32),
                    'Active_Time_Left': tf.FixedLenFeature([2], dtype=tf.float32),
                    'Product_Cost': tf.FixedLenFeature([2], dtype=tf.float32),
                    }

    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=None,
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    # default serving_input_receiver
    model.export_savedmodel("expected_models/exported_1", serving_input_receiver_fn, strip_default_attrs=True)

    # raw serving_input_receiver_fn
    # model.export_savedmodel("/expected_models/exported_1",
    #                         tf.estimator.export.build_raw_serving_input_receiver_fn(
    #                             features=feature_spec, default_batch_size=None),
    #                         strip_default_attrs=True)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
