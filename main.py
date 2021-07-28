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
from models import model
from utils import augment

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

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))


    if args.AUGMENT = True:
        train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
    else:
        train_dataset = train_dataset.shuffle(len(train_points)).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    model_structure = model.model_build(NUM_POINTS=args.NUM_POINTS, NUM_CLASSES=args.NUM_CLASSES, PRINT=args.PRINT)
    network = model_structure.load(args.MODEL)

    network.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=args.LEARNING_RATE),
        metrics=["sparse_categorical_accuracy"],
    )

    network.fit(train_dataset, epochs=args.EPOCHS, validation_data=test_dataset)


if __name__=="__main__":
    main()
