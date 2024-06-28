import numpy as np
import pandas as pd
import os
import yaml
import paths

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)

batch_size_training = config["learning_configuration"]["batch_size_training"]
batch_size_testing = config["learning_configuration"]["batch_size_testing"]
local_epochs = config["learning_configuration"]["local_epochs"]
FL = config["learning_configuration"]["FL"]
global_epochs = config["learning_configuration"]["global_epochs"]

if global_epochs <= 0:
    global_epochs = 1

if local_epochs <= 0:
    local_epochs = 1

if batch_size_training <= 0:
    batch_size_training = 1

if batch_size_testing <= 0:
    batch_size_testing = 1

# Fashion-MNIST has only 60000 training images
if batch_size_training > 60000:
    batch_size_training = 60000

# Fashion-MNIST has only 10000 testing images
if batch_size_testing > 10000:
    batch_size_testing = 10000

image_shape_training = (batch_size_training, 28, 28)
image_shape_testing = (batch_size_testing, 28, 28)


def prepare_train():
    df_train = pd.read_csv(str(paths.get_project_root()) + "\\fashion dataset\\fashion-mnist_train.csv")

    if FL:
        x_train = {}
        y_train = {}
        for i in range(local_epochs):
            sample = df_train.sample(n=batch_size_training, replace=False)
            sample = np.array(sample, dtype='float32')
            images = sample[:, 1:]
            labels = sample[:, 0]

            if image_shape_training is not None:
                images = images.reshape(image_shape_training)
            images = np.expand_dims(images, axis=1)

            x_train[str(i)] = images
            y_train[str(i)] = labels

    else:
        x_train = [[0] * local_epochs for i in range(global_epochs)]
        y_train = [[0] * local_epochs for i in range(global_epochs)]
        for i in range(global_epochs):
            for j in range(local_epochs):
                sample = df_train.sample(n=batch_size_training, replace=False)
                sample = np.array(sample, dtype='float32')
                images = sample[:, 1:]
                labels = sample[:, 0]

                if image_shape_training is not None:
                    images = images.reshape(image_shape_training)
                images = np.expand_dims(images, axis=1)

                x_train[i][j] = images
                y_train[i][j] = labels

    return x_train, y_train


def prepare_test():
    df_test = pd.read_csv(str(paths.get_project_root()) + "\\fashion dataset\\fashion-mnist_test.csv")

    x_test = [[0] * local_epochs for i in range(global_epochs)]
    y_test = [[0] * local_epochs for i in range(global_epochs)]
    for i in range(global_epochs):
        for j in range(local_epochs):
            sample = df_test.sample(n=batch_size_testing, replace=False)
            sample = np.array(sample, dtype='float32')
            images = sample[:, 1:]
            labels = sample[:, 0]

            if image_shape_testing is not None:
                images = images.reshape(image_shape_testing)
            images = np.expand_dims(images, axis=1)

            x_test[i][j] = images
            y_test[i][j] = labels

    return x_test, y_test
