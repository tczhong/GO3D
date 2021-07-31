import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

def save_prediction_image(model, test_dataset, class_map, log_dir):
    print('Saving prediction image...')

    data = test_dataset.take(1)

    points, labels = list(data)[0]
    points = points[:8, ...]
    labels = labels[:8, ...]

    # run test data through model
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)

    points = points.numpy()

    # plot points with predicted class and label
    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(
            "pred: {:}, label: {:}".format(
                class_map[preds[i].numpy()], class_map[labels.numpy()[i]]
            )
        )
        ax.set_axis_off()
    plt.savefig(log_dir+'prediction.png')
    plt.close()

def save_confusion_matrix(model, validation_points, validation_label, log_dir): 
    pred_scores = model.predict(validation_points)
    preds = tf.math.argmax(pred_scores, -1)
    mat = confusion_matrix(validation_label, preds)
    np.savetxt(log_dir+'confusion_matrix.txt', mat, fmt='%d', delimiter='\t')