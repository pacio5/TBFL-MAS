import numpy as np

from dataset import prepare_dataset
from learning import average_gradients, average_weights, calculate_metrics, predicting, training
from models import CNN, Personal
import paths
import torch
from torch import nn, Tensor
import unittest
import yaml

with open(str(paths.get_project_root()) + "\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)




class TestLearningAndModels(unittest.TestCase):
    def trend(self, metrics):
        trend = 0
        for i in range(1, len(metrics)):
            trend += metrics[i] - metrics[i-1]

        return trend/len(metrics)

    def test_CNN(self):
        all_labels = []
        all_predictions = []
        all_test_accuracies = []
        all_test_f1_scores = []
        all_test_precisions = []
        all_test_recalls = []
        cnn = CNN(10)
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)
        testing_losses = []
        training_losses = []

        for i in range(10):
            x_train, y_train, y_train_original_labels = prepare_dataset(
                config["learning_configuration"]["dataset_training"])
            x_test, y_test, y_test_original_labels = prepare_dataset(
                config["learning_configuration"]["dataset_testing"])

            training(criterion, device, cnn, optimizer, training_losses, x_train, y_train)

            predicting(all_labels, all_predictions, criterion, device, cnn, testing_losses,
                                    x_test, y_test_original_labels, y_test)

            calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                                    all_test_precisions, all_test_recalls)

        self.assertGreaterEqual(all_test_accuracies[-1], 0.5)
        self.assertGreaterEqual(all_test_f1_scores[-1], 0.5)
        self.assertGreaterEqual(all_test_precisions[-1], 0.5)
        self.assertGreaterEqual(all_test_recalls[-1], 0.5)
        self.assertGreaterEqual(self.trend(all_test_accuracies), 0.01)
        self.assertGreaterEqual(self.trend(all_test_f1_scores), 0.01)
        self.assertGreaterEqual(self.trend(all_test_precisions), 0.01)
        self.assertGreaterEqual(self.trend(all_test_recalls), 0.01)

    def test_Personal(self):
        all_labels = []
        all_predictions = []
        all_test_accuracies = []
        all_test_f1_scores = []
        all_test_precisions = []
        all_test_recalls = []
        personal = Personal(10)
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        optimizer = torch.optim.SGD(personal.parameters(), lr=0.1, momentum=0.9)
        testing_losses = []
        training_losses = []

        for i in range(10):
            x_train, y_train, y_train_original_labels = prepare_dataset(
                config["learning_configuration"]["dataset_training"])
            x_test, y_test, y_test_original_labels = prepare_dataset(
                config["learning_configuration"]["dataset_testing"])

            training(criterion, device, personal, optimizer, training_losses, x_train, y_train)

            predicting(all_labels, all_predictions, criterion, device, personal, testing_losses,
                       x_test, y_test_original_labels, y_test)

            calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                              all_test_precisions, all_test_recalls)

        print(all_test_accuracies)

        self.assertGreaterEqual(all_test_accuracies[-1], 0.5)
        self.assertGreaterEqual(all_test_f1_scores[-1], 0.5)
        self.assertGreaterEqual(all_test_precisions[-1], 0.5)
        self.assertGreaterEqual(all_test_recalls[-1], 0.5)
        self.assertGreaterEqual(self.trend(all_test_accuracies), 0.01)
        self.assertGreaterEqual(self.trend(all_test_f1_scores), 0.01)
        self.assertGreaterEqual(self.trend(all_test_precisions), 0.01)
        self.assertGreaterEqual(self.trend(all_test_recalls), 0.01)

    def control(self, avg, g1, g2):
        if isinstance(avg, list):
            for i in range(len(avg)):
                if isinstance(avg[i], list):
                    self.control(avg[i], g1[i], g2[i])
                else:
                    self.assertAlmostEqual(avg[i], (g1[i] + g2[i]) / 2)
        else:
            self.assertEqual(avg, (g1 + g2) / 2)

    def test_average_weights(self):
        avg = {}
        cnn1 = CNN(10)
        cnn2 = CNN(10)
        weights = {}
        weights["1"] = cnn1.state_dict()
        weights["2"] = cnn2.state_dict()

        average_weights(avg, weights)
        for i in avg["weights"].keys():
            list_avg = avg["weights"][i].tolist()
            list_g1 = weights["1"][i].tolist()
            list_g2 = weights["2"][i].tolist()
            self.control(list_avg, list_g1, list_g2)

        cnnAvg = CNN(10)
        cnnAvg.load_state_dict(avg["weights"])

    def test_average_gradients(self):
        cnn1 = CNN(10)
        cnn2 = CNN(10)
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        opt1 = torch.optim.SGD(cnn1.parameters(), lr=0.1, momentum=0.9)
        opt2 = torch.optim.SGD(cnn2.parameters(), lr=0.1, momentum=0.9)
        training_losses1 = []
        training_losses2 = []
        x_train, y_train, y_train_original_labels = prepare_dataset(
            config["learning_configuration"]["dataset_training"], batch_size=3000, local_epochs=10)

        training(criterion, device, cnn1, opt1, training_losses1, x_train, y_train)
        training(criterion, device, cnn2, opt2, training_losses2, x_train, y_train)

        avg = {}
        cnn3 = CNN(10)
        gradients = {}
        gradients["1"] = opt1.state_dict()
        gradients["2"] = opt2.state_dict()

        average_gradients(avg, gradients)

        for i in avg["gradients"]["state"]:
            for j in avg["gradients"]["state"][i].keys():
                list_avg = avg["gradients"]["state"][i][j].tolist()
                list_g1 = gradients["1"]["state"][i][j].tolist()
                list_g2 = gradients["2"]["state"][i][j].tolist()
                self.control(list_avg, list_g1, list_g2)

        optAvg = torch.optim.SGD(cnn3.parameters(), lr=0.1, momentum=0.9)
        before = optAvg.state_dict()["state"]
        optAvg.load_state_dict(avg["gradients"])
        after = optAvg.state_dict()["state"]
        self.assertNotEqual(after, before)


if __name__ == "__main__":
    unittest.main()