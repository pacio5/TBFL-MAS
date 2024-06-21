import numpy as np
import pandas as pd
import os

import torch


def walk_to_the_right_directory():
    # print("Current directory:", os.getcwd())
    os.chdir("../")
    # print("Current directory:", os.getcwd())


def prepare_train(batch_size, global_epochs, local_epochs, image_shape=None):
    if global_epochs <= 0:
        global_epochs = 1

    if local_epochs <= 0:
        local_epochs = 1

    if batch_size <= 0:
        batch_size = 1

    # Fashion-MNIST has only 60000 testing images
    if batch_size > 60000:
        batch_size = 60000

    df_train = pd.read_csv("fashion dataset/fashion-mnist_train.csv")

    X_train = [[0] * local_epochs for i in range(global_epochs)]
    Y_train = [[0] * local_epochs for i in range(global_epochs)]
    for i in range(global_epochs):
        for j in range(local_epochs):
            sample = df_train.sample(n=batch_size, replace=False)
            sample = np.array(sample, dtype='float32')
            images = sample[:, 1:]
            labels = sample[:, 0]

            if image_shape is not None:
                images = images.reshape(image_shape)
            images = np.expand_dims(images, axis=1)

            X_train[i][j] = images
            Y_train[i][j] = labels
    return X_train, Y_train


def prepare_test(batch_size, global_epochs, local_epochs, image_shape=None):
    if global_epochs <= 0:
        global_epochs = 1

    if local_epochs <= 0:
        local_epochs = 1

    if batch_size <= 0:
        batch_size = 1

    # Fashion-MNIST has only 10000 training images
    if batch_size > 10000:
        batch_size = 10000

    df_test = pd.read_csv("fashion dataset/fashion-mnist_test.csv")

    X_test = [[0] * local_epochs for i in range(global_epochs)]
    Y_test = [[0] * local_epochs for i in range(global_epochs)]
    for i in range(global_epochs):
        for j in range(local_epochs):
            sample = df_test.sample(n=batch_size, replace=False)
            sample = np.array(sample, dtype='float32')
            images = sample[:, 1:]
            labels = sample[:, 0]

            if image_shape is not None:
                images = images.reshape(image_shape)
            images = np.expand_dims(images, axis=1)

            X_test[i][j] = images
            Y_test[i][j] = labels
    return X_test, Y_test
