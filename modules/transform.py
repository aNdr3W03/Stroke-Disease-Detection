"""
Author: Andrew Benedictus Jamesie
Date: 20/01/2023
This is the transform.py module.
Usage:
- Transform module
"""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    'gender': 2,
    'ever_married': 2,
    'work_type': 5,
    'Residence_type': 2,
    'smoking_status': 4
}

NUMERICAL_FEATURES = [
    'age',
    'hypertension',
    'heart_disease',
    'avg_glucose_level',
    'bmi'
]

LABEL_KEY = 'stroke'


def transformed_name(key):
    """Renaming transformed features

    Args:
        key (str): the key to be transformed

    Returns:
        str: transformed key
    """
    return key + "_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Convert a label (0 or 1) into a one-hot vector

    Args:
        label_tensor (int): label tensor (0 or 1)
        num_labels (int, optional): num of label, defaults to 2

    Returns:
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features

    Args:
        inputs (dict): map from feature keys to raw features

    Returns:
        outputs (dict): map from feature keys to transformed features
    """
    outputs = {}

    for keys, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            inputs[keys], top_k=dim+1
        )
        outputs[transformed_name(keys)] = convert_num_to_one_hot(
            int_value, num_labels=dim+1
        )

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
