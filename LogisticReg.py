import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from utils import preload_model

filename = 'finalized_model.sav'


def logistic_regression(custom_train_ds, classes):

    # create datasets
    data = []
    labels = []
    for file in custom_train_ds:
        image = cv2.imread(str(file))
        label = str(file.parent).split('/')[-1]
        data.append(image.reshape(-1)/.255)
        labels.append(classes.index(label))

    x_train, x_test, y_train, y_test = train_test_split(np.array(data, np.float32), np.array(labels, np.float32),
                                                        test_size=0.3, random_state=16)

    logisticRegr = preload_model(filename)

    # if not logisticRegr:
    #     logisticRegr = LogisticRegression(solver='lbfgs', max_iter=100)
    #
    #     logisticRegr.fit(x_train, y_train)
    #
    #     pickle.dump(logisticRegr, open(filename, 'wb'))

    predict = logisticRegr.predict(x_test)
    print(predict)

    train_sizes, train_scores, test_scores = learning_curve(logisticRegr, x_train, y_train, train_sizes=[50, 100, 1000, 10000, 100000])
    print(train_sizes, train_scores, test_scores)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--',
             label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()