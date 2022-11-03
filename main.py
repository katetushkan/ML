import random
import shutil

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import pickle

from LogisticReg import logistic_regression
from neuronNetwork import neural_network
from utils import shuffle_files, compute_hashes, remove_hashes, open_image, is_balanced

CLASSES_COUNT = 10
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def classify(method):
    print(tf.__version__)

    # dataset import
    import pathlib
    dataset_url = 'https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz'
    data_dir = tf.keras.utils.get_file(
        "notMNIST_large",
        origin=dataset_url,
        untar=True
    )
    data_dir = pathlib.Path(data_dir)

    # open some images from the given dataset
    for index, char in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'J', 'H', 'I', 'J']):
        open_image(data_dir, char, index + 1)

    # check if data is balanced
    balanced, stat = is_balanced(data_dir, CLASSES, CLASSES_COUNT)
    print(f"Balanced score: {balanced}, Balance stat: {stat}")

    dataset_with_dupes = shuffle_files(data_dir)

    # remove duplicates from training
    hashes = compute_hashes(dataset_with_dupes)
    custom_train_ds = remove_hashes(hashes)

    # magic is happening here
    if method == 'nn':
        neural_network(custom_train_ds, CLASSES)
    else:
        logistic_regression(custom_train_ds, CLASSES)


if __name__ == '__main__':
    classify('')
