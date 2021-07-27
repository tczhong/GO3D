import os
import glob
import trimesh
import numpy as np
import pickle as pkl
import tensorflow as tf

def download_data():
    data_dir = tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(data_dir), "ModelNet10")
    return data_dir

def parse_dataset(data_dir, num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

def main():
    NUM_POINTS = 2048

    data_dir = download_data()

    print("Parsing data...")
    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        data_dir, NUM_POINTS
    )

    with open('parsed_data.pkl', 'wb') as outfile:
        pkl.dump((train_points, test_points, train_labels, test_labels, CLASS_MAP), outfile, pkl.HIGHEST_PROTOCOL)
    
if __name__=="__main__":
    main()
