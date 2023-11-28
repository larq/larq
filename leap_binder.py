from typing import List

import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from larq import (
    activations,
    callbacks,
    constraints,
    context,
    layers,
    math,
    metrics,
    models,
    optimizers,
    quantizers,
    utils,
)
from larq.layers import QuantDense
import tensorflow as tf


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    train_X = np.random.uniform(0, 1, (100, 512))

    val_X = np.random.uniform(0, 1, (50, 512))

    train = PreprocessResponse(length=len(train_X), data=train_X)
    val = PreprocessResponse(length=len(val_X), data=val_X)
    response = [train, val]
    return response


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image. 
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data[idx].astype('float32')


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return np.random.randint(0, 2)


def placeholder_loss(y_true, y_pred: tf.Tensor, **kwargs) -> tf.Tensor:
    return tf.reduce_mean(y_true, axis=-1) * 0


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_encoder, name='input')
leap_binder.set_ground_truth(function=gt_encoder, name='gt')
leap_binder.set_custom_layer(QuantDense, 'QuantDense')
leap_binder.add_custom_loss(placeholder_loss, 'zero_loss')

if __name__ == '__main__':
    leap_binder.check()
