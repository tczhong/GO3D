import os
import glob
import trimesh
import numpy as np
import pickle as pkl
import tensorflow as tf
from matplotlib import pyplot as plt

def download_data():
    data_dir = tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
        cache_dir='./data/'
    )
    data_dir = os.path.join(os.path.dirname(data_dir), "ModelNet10")
    print("Saving file to directory: ", data_dir)
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

def save_image(class_map, data_dir='./data/datasets/ModelNet10', num_points=2048, num_per_obj=3):
    file_dir = "./data/images/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    for key, obj in class_map:
        print("Saving images for", obj)
        for i in range(1, num_per_obj+1):
            file_name = obj + "_000" + str(i)
            mesh = trimesh.load(os.path.join(data_dir, obj + "/train/" + file_name + ".off"))
            points = mesh.sample(num_points)

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])
            ax.set_axis_off()
            plt.savefig(file_dir+file_name+'.png')
            plt.close()

def save_data(num_points = 2048):
    print('Downloading data...')
    data_dir = download_data()

    print('Parsing data...')
    train_points, test_points, train_labels, test_labels, class_map = parse_dataset(
        data_dir, num_points
    )

    print('Class map:', class_map)

    with open('./data/parsed_data_' + str(num_points) + '.pkl', 'wb') as outfile:
        pkl.dump((train_points, test_points, train_labels, test_labels, class_map), outfile, pkl.HIGHEST_PROTOCOL)

    save_image(class_map, data_dir, num_points)


if __name__=="__main__":
    save_data()
