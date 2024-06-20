import pandas as pd
import os


def walk_to_the_right_directory():
    # print("Current directory:", os.getcwd())
    os.chdir("../")
    # print("Current directory:", os.getcwd())


def prepare_train(batch_size, epoch):
    if epoch <= 0:
        epoch = 1

    if batch_size <= 0:
        batch_size = 1

    # Fashion-MNIST has only 60000 testing images
    if batch_size > 60000:
        batch_size = 60000

    df_train = pd.read_csv("fashion dataset/fashion-mnist_train.csv")

    X_train = []
    Y_train = []
    for i in range(epoch):
        epoch = df_train.sample(n=batch_size, replace=False)
        X_train.append(epoch.drop(["label"], axis=1).to_numpy())
        Y_train.append(epoch.label.values)

    return X_train, Y_train


def prepare_test(batch_size, epoch):
    if epoch <= 0:
        epoch = 1

    if batch_size <= 0:
        batch_size = 1

    # Fashion-MNIST has only 10000 training images
    if batch_size > 10000:
        batch_size = 10000

    df_test = pd.read_csv("fashion dataset/fashion-mnist_test.csv")

    X_test = []
    Y_test = []
    for i in range(epoch):
        epoch = df_test.sample(n=batch_size, replace=False, random_state=1)
        X_test.append(epoch.drop(["label"], axis=1).to_numpy())
        Y_test.append(epoch.label.values)

    return X_test, Y_test
