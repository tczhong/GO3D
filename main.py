import os
import glob
import trimesh
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import yaml
from data import data
from models import model
from utils import augment
from datetime import datetime

parser = argparse.ArgumentParser(description='Go3D')
parser.add_argument('--config', default='./configs/config_pointnet.yaml')

tf.random.set_seed(42)


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k,v in config[key].items():
            setattr(args, k, v)

    # Load Data

    file_name = './data/parsed_data.pkl'
    train_points = None
    test_points = None
    train_labels = None
    test_labels = None
    class_map = None

    if not os.path.isfile(file_name):
        data.save_data(args.BATCH_SIZE)

    with open(file_name, 'rb') as infile:
        train_points, test_points, train_labels, test_labels, class_map = pkl.load(infile)
        print('class_map:', class_map)
        print('train_points:', train_points.shape)
        print('test_points:', test_points.shape)
        print('train_labels:', train_labels.shape)
        print('test_labels:', test_labels.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))


    if args.AUGMENT == True:
        train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(args.BATCH_SIZE)
    else:
        train_dataset = train_dataset.shuffle(len(train_points)).batch(args.BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(args.BATCH_SIZE)

    # Construct Model
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './outputs/'+args.MODEL+'_'+time_stamp+'/'
    os.mkdir(log_dir)
    print('MODEL LOGGGER:', log_dir)

    model_structure = model.model_build(NUM_POINTS=args.NUM_POINTS, NUM_CLASSES=args.NUM_CLASSES,
                                        DROPOUT_RATE=args.DROPOUT_RATE, PRINT=args.PRINT)
    network = model_structure.load(MODEL=args.MODEL, log_dir=log_dir)

    network.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=args.LEARNING_RATE),
        metrics=["sparse_categorical_accuracy"],
    )

    model_callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir),
        #keras.callbacks.ModelCheckpoint(filepath=log_dir+args.MODEL+'_{epoch:02d}.h5'),
        keras.callbacks.CSVLogger(filename=log_dir+'model_training.log')
    ]

    network.fit(train_dataset, epochs=args.EPOCHS, validation_data=test_dataset,
                callbacks=model_callbacks)

if __name__=="__main__":
    main()
