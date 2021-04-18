import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    #Place holder and padding
    X_input = layers.Input(shape=input_shape)
    X = layers.ZeroPadding2D(padding=(3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), name='Conv_0')(X)
    X = layers.BatchNormalization(axis=3, name='bn_0')(X)
    X = layers.Activation(activation='relu')(X)

    X = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_0')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(units=1, activation='sigmoid', name='fc_0')(X)

    model = models.Model(inputs=X_input, outputs=X, name='Happy_Model')
    return model





