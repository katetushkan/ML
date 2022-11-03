import pickle
import random

import PIL
import cv2
import tensorflow as tf


def is_balanced(directory, labels, label_count):
    balanced_stat = {}
    for label in labels:
        example_count = len(list(directory.glob(f"{label}/*.png")))
        balanced_stat[label] = example_count

    max_count = 0
    sum_count = 0
    for value in balanced_stat.values():
        if value > max_count:
            max_count = value
        sum_count += value

    average = sum_count / label_count
    balanced = average / max_count
    return [balanced, balanced_stat]


def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image


def open_image(directory, label, count):
    images = list(directory.glob(f"{label}/*.png"));
    with PIL.Image.open(images[count]) as img:
        img.show()


def shuffle_files(directory):
    filenames = list(directory.glob("*/*.png"))
    filenames.sort()
    random.seed(230)
    random.shuffle(filenames)

    return filenames


def dhash(image, hashSize=8):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient

    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def compute_hashes(dataset):
    image_paths = dataset
    hashes = {}

    # loop over image paths
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is not None:
            if image.shape[0] != 0 and image.shape[1] != 0:
                h = dhash(image)

                existing_images = hashes.get(h, [])
                existing_images.append(image_path)

                hashes[h] = existing_images

    return hashes


def remove_hashes(hashes, remove=False):
    new_train_ds = []
    hash_len = 0
    for (h, hashes_paths) in hashes.items():
        new_train_ds.append(hashes_paths[0])
        hash_len += len(hashes_paths)

    print(f"{hash_len - len(new_train_ds)} duplicates was deleted")

    return new_train_ds


def split_ds_nn(dataset, train_count, val_count):
    train_ds = dataset.take(train_count).batch(32)
    val_ds = dataset.skip(train_count).take(val_count).batch(32)
    control_ds = dataset.skip(train_count + val_count).batch(32)

    return [train_ds, val_ds, control_ds]


def preload_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
