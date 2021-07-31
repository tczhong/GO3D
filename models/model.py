import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.components import conv_bn, dense_bn, OrthogonalRegularizer, tnet, tnet_full
import numpy as np
from datetime import datetime

class model_build():

    def __init__(self, NUM_POINTS, NUM_CLASSES, PRINT, DROPOUT_RATE):
        self.NUM_POINTS = NUM_POINTS
        self.NUM_CLASSES = NUM_CLASSES
        self.PRINT = PRINT
        self.DROPOUT_RATE = DROPOUT_RATE

    def pointnet_mod(self):
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
        x = layers.Dropout(self.DROPOUT_RATE)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(self.DROPOUT_RATE)(x)

        outputs = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)

        network = keras.Model(inputs=inputs, outputs=outputs, name="pointnet_mod")

        return network

    def pointnet(self):
        inputs = keras.Input(shape=(self.NUM_POINTS, 3))

        x = tnet_full(inputs, 3)
        x = conv_bn(x, 64)
        x = conv_bn(x, 64)
        x = tnet_full(x, 64)
        x = conv_bn(x, 64)
        x = conv_bn(x, 128)
        x = conv_bn(x, 1024)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 512)
        x = layers.Dropout(self.DROPOUT_RATE)(x)
        x = dense_bn(x, 256)
        x = layers.Dropout(self.DROPOUT_RATE)(x)

        outputs = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)

        network = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

        return network


    def load(self, MODEL, log_dir):
        if MODEL == 'pointnet':
            network = self.pointnet()
        elif MODEL == 'pointnet_mod':
            network = self.pointnet_mod()
        else:
            print('Invalid MODEL...')

        if self.PRINT == True:
            print(network.summary)

        with open(log_dir+MODEL+'_model_summary.txt', 'w') as fh:
            network.summary(print_fn=lambda x: fh.write(x + '\n'))

        return network
