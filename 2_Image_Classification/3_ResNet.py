# Image classification with Data augmentation, rescaling, convolutional layer, pooling layer, drop out etc.
# =============================================
# API doc from TensorFlow Org: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU
# Reference of convolutional layer, pooling layer can be found at《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》- Chapter 14

# Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Resnet unit built with Keras
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='Relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=True),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=True),
            keras.layers.BatchNormalization(),
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=True),
                keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z)


# Build Resnet-34 model with Kera sequential API
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding='same', use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
pre_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == pre_filters else 2
    model.add(ResidualUnit(filters=filters, strides=strides, activation='relu'))
    pre_filters = filters
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
