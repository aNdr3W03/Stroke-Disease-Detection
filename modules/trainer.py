"""
Author: Andrew Benedictus Jamesie
Date: 20/01/2023
This is the trainer.py module.
Usage:
- Training module
"""

import os

import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from keras.utils.vis_utils import plot_model
from transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)
from tuner import input_fn


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example"""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature"""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def get_model(hyperparameters, show_summary=True):
    """This function defines a Keras model

    Args:
        hyperparameters (kt.HyperParameters): hyperparameters setting
        show_summary (bool): show model summary, defaults to True

    Returns:
        tf.keras.Model: model as a keras object
    """
    input_features = []

    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim+1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = layers.concatenate(input_features)
    deep = layers.Dense(
        hyperparameters['dense_units'], activation='relu')(concatenate)

    for _ in range(hyperparameters['num_layers']):
        deep = layers.Dense(
            hyperparameters['dense_units'], activation='relu')(deep)

    deep = layers.Dropout(hyperparameters['dropout_rate'])(deep)
    outputs = layers.Dense(1, activation='sigmoid')(deep)
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyperparameters['learning_rate']
        ),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    if show_summary:
        model.summary()

    return model


def run_fn(fn_args):
    """Train the model based on given args

    Args:
        fn_args: holds args used to train the model as name/value pairs
    """
    hyperparameters = fn_args.hyperparameters['values']
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    model = get_model(hyperparameters)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq='batch'
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        patience=10
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    callbacks = [
        tensorboard_callback,
        early_stop_callback,
        model_checkpoint_callback
    ]

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        epochs=hyperparameters['tuner/initial_epoch'],
        verbose=1
    )

    signatures = {
        'serving_default': get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        ),
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )

    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )
