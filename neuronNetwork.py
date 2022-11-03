import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

from utils import split_ds_nn


def neural_network(custom_train_ds, classes):

    data = []
    labels = []
    for file in custom_train_ds:
        image = cv2.imread(str(file))
        label = str(file.parent).split('/')[-1]
        data.append(image)
        labels.append(classes.index(label))

    images_ds = tf.data.Dataset.from_tensor_slices(data)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images_ds, labels_ds))

    train_ds, val_ds, control_ds = split_ds_nn(dataset, 200000, 19000)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(labels))
    ])

    # print(model.summary())

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=1
    )

    loss, accuracy = model.evaluate(control_ds)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(5)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


    # print(f"Control dataset: loss - {loss}, accuracy - {accuracy}")