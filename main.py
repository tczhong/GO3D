import os
import glob
import trimesh
import pickle as pkl
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import argparse
import yaml
from models import model_build

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k,v in config[key].items():
            setattr(args, k, v)
    
    file_name = './data/parsed_data.pkl'
    train_points = None
    test_points = None
    train_labels = None
    test_labels = None
    class_map = None

    if not os.path.isfile(file_name):
        print(file_name, ' does not exist.\n \
              Please run ./data/data.py to downlaod the data.')

    with open(file_name, 'rb') as infile:
        train_points, test_points, train_labels, test_labels, class_map = pkl.load(infile)
        print('class_map:', class_map)
        print('train_points:', train_points.shape)
        print('test_points:', test_points.shape)
        print('train_labels:', train_labels.shape)
        print('test_labels:', test_labels.shape)

    model_build(NUM_POINTS=args.NUM_POINTS, NUM_CLASSES=args.NUM_CLASSES, PRINT=args.PRINT)
    model = model_build.load(args.MODEL)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=args.LEARNING_RATE),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(train_dataset, epochs=args.EPOCHS, validation_data=test_dataset)


if __name__=="__main__":
    main()
