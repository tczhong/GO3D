import os
import glob
import trimesh
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
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

if __name__=="__main__":
    main()
