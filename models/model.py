import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.components import conv_bn, dense_bn, OrthogonalRegularizer, tnet
import numpy as np

class model_build():

    def __init__(self, NUM_POINTS, NUM_CLASSES, PRINT):
        self.NUM_POINTS = NUM_POINTS
        self.NUM_CLASSES = NUM_CLASSES
        self.PRINT = PRINT

    def pointnet(self):
        inputs = keras.Input(shape=(self.NUM_POINTS, 3))

        x = tnet(inputs, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)

        network = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

        return network

    def load(self, MODEL):
        if MODEL == 'pointnet':
            network = self.pointnet()

        if self.PRINT == True:
            print(network.summary)

        return network