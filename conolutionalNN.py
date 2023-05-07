import os
import tarfile
import urllib

import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import models


def partial_linear(x):
    return tf.maximum(x, 0.1 * x)


# Define a function to download and extract the dataset
def download_data(url, local_dir):
    response = requests.get(url)
    compressed_file = BytesIO(response.content)
    tar_file = tarfile.open(fileobj=compressed_file, mode='r|gz')
    tar_file.extractall(local_dir)
    tar_file.close()


# Normalize the images
def normalize(data):
    return (data.astype(np.float32) - 128.0) / 128.0


# Remove duplicates from the dataset
def remove_duplicates(data):
    _, index = np.unique(data[:, 1:], axis=0, return_index=True)
    return data[index]


def load_data():
    data = []
    labels = []
    for label, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
        folder = os.path.join('notMNIST_large/notMNIST_large', letter)
        for file in os.listdir(folder):
            try:
                with Image.open(os.path.join(folder, file)) as img:
                    data.append(np.array(img))
                    labels.append(label)
            except:
                pass
    data = np.array(data)
    labels = np.array(labels)
    data = data.reshape((-1, 28, 28, 1))
    data = remove_duplicates(np.hstack([labels.reshape(-1, 1), data.reshape(-1, 28 * 28)]))

    train_data = data[:50000]
    test_data = data[50000:]
    train_images = np.array([x[0] for x in train_data]).reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    train_labels = np.array([x[1] for x in train_data])
    test_images = np.array([x[0] for x in test_data]).reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_labels = np.array([x[1] for x in test_data])

    return train_images, train_labels, test_images, test_labels


def conv_neural_network():

    # Download the dataset
    url = 'https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz'
    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

    # Extract the dataset
    if not os.path.exists('notMNIST_large'):
        with tarfile.open(filename) as tar:
            tar.extractall()

    # Load the data
    x_train, y_train, x_test, y_test = load_data()

    # Normalize the images
    x_train = normalize(x_train)

    # Create the convolutional neural network
    model = models.Sequential([
        # First Convolutional Layer
        keras.layers.Conv2D(32, (3, 3), activation=partial_linear, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),

        # Second Convolutional Layer
        keras.layers.Conv2D(64, (3, 3), activation=partial_linear),
        keras.layers.MaxPooling2D((2, 2)),

        # Flattening Layer
        keras.layers.Flatten(),

        # First Fully Connected Layer
        keras.layers.Dense(64, activation='relu'),

        # Output Layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # Define the model architecture for the 2nd task
    # model = tf.keras.Sequential([
    #     # First pooling layer
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), input_shape=(28, 28, 1)),
    #
    #     # Second pooling layer
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #
    #     # Flatten the output from the pooling layers
    #     tf.keras.layers.Flatten(),
    #
    #     # Dense layer for classification
    #     tf.keras.layers.Dense(units=128, activation='relu'),
    #
    #     # Output Layer
    #     tf.keras.layers.Dense(units=10, activation='softmax')
    # ])

    # Implementation of the LeNet-5 architecture
    # model = tf.keras.Sequential([
    #     # First Convolutional Layer
    #     keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),

    #     # Second Convolutional Layer
    #     keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),

    #     # Flattening Layer
    #     keras.layers.Flatten(),

    #     # First Fully Connected Layer
    #     keras.layers.Dense(units=120, activation='relu'),

    #     # Second Fully Connected Layer
    #     keras.layers.Dense(units=84, activation='relu'),

    #     # Output Layer
    #     keras.layers.Dense(units=10, activation='softmax')
    # ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64)

    # Evaluate the model
    _, test_acc = model.evaluate(x_train, y_train, verbose=0)
    print('Test accuracy:', test_acc)

    # Print the summary of the model
    model.summary()
