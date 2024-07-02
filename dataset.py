import numpy as np
import pandas as pd
import paths
import yaml

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)


def prepare_dataset(path_to_dataset_from_project_root, classes_of_data_object=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 batch_size=3000, local_epochs=5):
    if local_epochs <= 0:
        local_epochs = 1

    if batch_size <= 0:
        batch_size = 1

    # Fashion-MNIST has only 10000 testing images
    if batch_size > 10000:
        batch_size = 10000

    image_shape_testing = (batch_size, 28, 28)
    dataframe = pd.read_csv(str(paths.get_project_root()) + path_to_dataset_from_project_root)
    dataframe = dataframe[dataframe['label'].isin(classes_of_data_object)]

    x_test = {}
    y_test = {}
    y_labels = {}
    for i in range(local_epochs):
        sample = dataframe.sample(n=batch_size, replace=False)
        sample = np.array(sample, dtype='float32')
        images = sample[:, 1:]
        labels = sample[:, 0]

        if image_shape_testing is not None:
            images = images.reshape(image_shape_testing)
        images = np.expand_dims(images, axis=1)

        labels_reshape = np.zeros((batch_size, 10), dtype='float32')

        for b in range(batch_size):
            labels_reshape[b, int(labels[b])] = 1

        x_test[str(i)] = images
        y_test[str(i)] = labels_reshape
        y_labels[str(i)] = labels
    return x_test, y_test, y_labels
