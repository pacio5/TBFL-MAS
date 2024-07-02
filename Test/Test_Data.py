from dataset import prepare_dataset
import numpy as np
import paths
import unittest
import yaml

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)


class TestDataset(unittest.TestCase):

    def test_prepare_dataset(self):
        path_to_dataset_form_project_root = "\\fashion dataset\\fashion-mnist_test.csv"
        classes = [1, 5, 8]
        x_train, y_train, y_original_labels = prepare_dataset(path_to_dataset_form_project_root, classes, 200, 10)
        self.assertEqual(10, len(x_train))
        self.assertEqual((200, 1, 28, 28), x_train["0"].shape)
        self.assertEqual((200, 10), y_train["0"].shape)
        self.assertEqual(classes, sorted(np.unique(y_original_labels["0"])))


if __name__ == "__main__":
    unittest.main()

