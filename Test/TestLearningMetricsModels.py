from Utilities.Argparser import Argparser
from Utilities.Data import Data
from Utilities.Learning import Learning
from Utilities.Metrics import Metrics
from Utilities.Models import CNN, PersonalCNN
import os
from Utilities.Paths import Paths
import torch
from torch import nn
import unittest
from unittest import mock

torch.manual_seed(0)


class TestLearningMetricsModels(unittest.TestCase):
    def test_CNN(self):
        # setting variables
        all_labels = []
        all_predictions = []
        all_test_accuracies = {}
        all_test_f1_scores = {}
        all_test_precisions = {}
        all_test_recalls = {}
        args = Argparser.args_parser()
        epoch = 25
        f1_scores_per_classes = {}
        for i in range(10):
            f1_scores_per_classes[str(i)] = {}
        precisions_per_classes = {}
        for i in range(10):
            precisions_per_classes[str(i)] = {}
        recalls_per_classes = {}
        for i in range(10):
            recalls_per_classes[str(i)] = {}
        cnn = CNN(10)
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)
        testing_losses = []
        training_losses = []

        # train, predict and calculate metrics
        for i in range(epoch):
            x_train, y_train, y_train_original_labels = Data.prepare_dataset(args.dataset_training)
            x_test, y_test, y_test_original_labels = Data.prepare_dataset(args.dataset_testing)

            Learning.training(criterion, device, cnn, optimizer, training_losses, x_train, y_train)

            Learning.predicting(all_labels, all_predictions, criterion, device, cnn, testing_losses,
                                x_test, y_test_original_labels, y_test)

            Metrics.calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                                      all_test_precisions, all_test_recalls, i + 1)

            Metrics.calculate_f1_score_per_classes(all_labels, all_predictions, i + 1, f1_scores_per_classes)

            Metrics.calculate_precisions_per_classes(all_labels, all_predictions, i + 1, precisions_per_classes)

            Metrics.calculate_recalls_per_classes(all_labels, all_predictions, i + 1, recalls_per_classes)

        # assert performance of the CNN model
        self.assertGreaterEqual(all_test_accuracies[str(epoch)], 0.8)
        self.assertGreaterEqual(all_test_f1_scores[str(epoch)], 0.8)
        self.assertGreaterEqual(all_test_precisions[str(epoch)], 0.8)
        self.assertGreaterEqual(all_test_recalls[str(epoch)], 0.8)

        for i in f1_scores_per_classes.keys():
            self.assertGreaterEqual(f1_scores_per_classes[i][str(epoch)], 0.4)
        for i in precisions_per_classes.keys():
            self.assertGreaterEqual(precisions_per_classes[i][str(epoch)], 0.4)
        for i in recalls_per_classes.keys():
            self.assertGreaterEqual(recalls_per_classes[i][str(epoch)], 0.4)

    def test_Personal(self):
        # setting variables
        all_labels = []
        all_predictions = []
        all_test_accuracies = {}
        all_test_f1_scores = {}
        all_test_precisions = {}
        all_test_recalls = {}
        args = Argparser.args_parser()
        epoch = 25
        f1_scores_per_classes = {}
        for i in range(10):
            f1_scores_per_classes[str(i)] = {}
        precisions_per_classes = {}
        for i in range(10):
            precisions_per_classes[str(i)] = {}
        recalls_per_classes = {}
        for i in range(10):
            recalls_per_classes[str(i)] = {}
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        personal = PersonalCNN(10)
        optimizer = torch.optim.SGD(personal.parameters(), lr=0.1, momentum=0.9)
        testing_losses = []
        training_losses = []

        # train, predict and calculate metrics
        for i in range(epoch):
            x_train, y_train, y_train_original_labels = Data.prepare_dataset(args.dataset_training)
            x_test, y_test, y_test_original_labels = Data.prepare_dataset(args.dataset_testing)

            Learning.training(criterion, device, personal, optimizer, training_losses, x_train, y_train)

            Learning.predicting(all_labels, all_predictions, criterion, device, personal, testing_losses,
                                x_test, y_test_original_labels, y_test)

            Metrics.calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                                      all_test_precisions, all_test_recalls, i + 1)

            Metrics.calculate_f1_score_per_classes(all_labels, all_predictions, i + 1, f1_scores_per_classes)

            Metrics.calculate_precisions_per_classes(all_labels, all_predictions, i + 1, precisions_per_classes)

            Metrics.calculate_recalls_per_classes(all_labels, all_predictions, i + 1, recalls_per_classes)

        # assert performance of the Personal model
        self.assertGreaterEqual(all_test_accuracies[str(epoch)], 0.8)
        self.assertGreaterEqual(all_test_f1_scores[str(epoch)], 0.8)
        self.assertGreaterEqual(all_test_precisions[str(epoch)], 0.8)
        self.assertGreaterEqual(all_test_recalls[str(epoch)], 0.8)

        for i in f1_scores_per_classes.keys():
            self.assertGreaterEqual(f1_scores_per_classes[i][str(epoch)], 0.4)
        for i in precisions_per_classes.keys():
            self.assertGreaterEqual(precisions_per_classes[i][str(epoch)], 0.4)
        for i in recalls_per_classes.keys():
            self.assertGreaterEqual(recalls_per_classes[i][str(epoch)], 0.4)

    def test_reproducibility_CNN(self):
        # setting variables
        all_labels = []
        all_predictions = []
        all_test_accuracies = {}
        all_test_f1_scores = {}
        all_test_precisions = {}
        all_test_recalls = {}
        args = Argparser.args_parser()
        cnn = CNN(10)
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)
        testing_losses = []
        training_losses = []

        # train, predict and calculate metrics
        for i in range(3):
            x_train, y_train, y_train_original_labels = Data.prepare_dataset(args.dataset_training)
            x_test, y_test, y_test_original_labels = Data.prepare_dataset(args.dataset_testing)

            Learning.training(criterion, device, cnn, optimizer, training_losses, x_train, y_train)

            Learning.predicting(all_labels, all_predictions, criterion, device, cnn, testing_losses,
                                x_test, y_test_original_labels, y_test)

            Metrics.calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                                      all_test_precisions, all_test_recalls, i + 1)

        # test reproducibility
        accuracies = {'1': 0.5606, '2': 0.6030666666666666, '3': 0.6477777777777778}
        f1_scores = {'1': 0.47786979113867517, '2': 0.5563307022876182, '3': 0.612806950726641}
        precisions = {'1': 0.5629303764428463, '2': 0.6519550022587194, '3': 0.6605772201264324}
        recalls = {'1': 0.5606, '2': 0.6030666666666666, '3': 0.6477777777777778}
        self.assertDictEqual(accuracies, all_test_accuracies)
        self.assertDictEqual(f1_scores, all_test_f1_scores)
        self.assertDictEqual(precisions, all_test_precisions)
        self.assertDictEqual(recalls, all_test_recalls)

    def test_reproducibility_Personal(self):
        # setting variables
        all_labels = []
        all_predictions = []
        all_test_accuracies = {}
        all_test_f1_scores = {}
        all_test_precisions = {}
        all_test_recalls = {}
        args = Argparser.args_parser()
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        personal = PersonalCNN(10)
        optimizer = torch.optim.SGD(personal.parameters(), lr=0.1, momentum=0.9)
        testing_losses = []
        training_losses = []

        # train, predict and calculate metrics
        for i in range(3):
            x_train, y_train, y_train_original_labels = Data.prepare_dataset(args.dataset_training)
            x_test, y_test, y_test_original_labels = Data.prepare_dataset(args.dataset_testing)

            Learning.training(criterion, device, personal, optimizer, training_losses, x_train, y_train)

            Learning.predicting(all_labels, all_predictions, criterion, device, personal, testing_losses,
                                x_test, y_test_original_labels, y_test)

            Metrics.calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                                      all_test_precisions, all_test_recalls, i + 1)

        # test reproducibility
        accuracies = {'1': 0.6084, '2': 0.6589333333333334, '3': 0.6877333333333333}
        f1_scores = {'1': 0.5390148492907642, '2': 0.6121734715271161, '3': 0.6487880243509141}
        precisions = {'1': 0.6334618625982138, '2': 0.6708476329098448, '3': 0.6895511303854079}
        recalls = {'1': 0.6084, '2': 0.6589333333333334, '3': 0.6877333333333333}

        self.assertDictEqual(accuracies, all_test_accuracies)
        self.assertDictEqual(f1_scores, all_test_f1_scores)
        self.assertDictEqual(precisions, all_test_precisions)
        self.assertDictEqual(recalls, all_test_recalls)

    def control(self, avg, g1, g2):
        # assert if the weights or gradients are almost equal
        if isinstance(avg, list):
            for i in range(len(avg)):
                if isinstance(avg[i], list):
                    self.control(avg[i], g1[i], g2[i])
                else:
                    self.assertAlmostEqual((g1[i] + g2[i]) / 2, avg[i])
        else:
            self.assertAlmostEqual((g1 + g2) / 2, avg)

    def test_store_and_plot_metrics(self):
        # setting variables
        agent_name = "test"
        all_labels = []
        all_predictions = []
        all_test_accuracies = {}
        all_test_f1_scores = {}
        all_test_precisions = {}
        all_test_recalls = {}
        all_testing_losses = {}
        all_training_losses = {}
        batch_sizes_per_classes = {'0': 300, '1': 300, '2': 300, '3': 300, '4': 300, '5': 300, '6': 300, '7': 300,
                                   '8': 300, '9': 300}
        f1_scores_per_classes = {}
        for i in range(10):
            f1_scores_per_classes[str(i)] = {}
        precisions_per_classes = {}
        for i in range(10):
            precisions_per_classes[str(i)] = {}
        recalls_per_classes = {}
        for i in range(10):
            recalls_per_classes[str(i)] = {}
        args = Argparser.args_parser()
        cnn = CNN(10)
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)

        # train, predict and calculate metrics
        for i in range(3):
            x_train, y_train, y_train_original_labels = Data.prepare_dataset(args.dataset_training)
            x_test, y_test, y_test_original_labels = Data.prepare_dataset(args.dataset_testing)
            testing_losses = []
            training_losses = []

            Learning.training(criterion, device, cnn, optimizer, training_losses, x_train, y_train)

            all_training_losses[str(i)] = sum(training_losses) / len(training_losses)

            Learning.predicting(all_labels, all_predictions, criterion, device, cnn, testing_losses,
                                x_test, y_test_original_labels, y_test)

            all_testing_losses[str(i)] = sum(testing_losses) / len(testing_losses)

            Metrics.calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                                      all_test_precisions, all_test_recalls, i + 1)

            Metrics.calculate_f1_score_per_classes(all_labels, all_predictions, i + 1, f1_scores_per_classes)

            Metrics.calculate_precisions_per_classes(all_labels, all_predictions, i + 1, precisions_per_classes)

            Metrics.calculate_recalls_per_classes(all_labels, all_predictions, i + 1, recalls_per_classes)

        # store the metrics
        Metrics.store_metrics(agent_name, all_test_accuracies, all_test_f1_scores, all_test_precisions,
                              all_test_recalls,
                              all_testing_losses, all_training_losses, args, batch_sizes_per_classes,
                              f1_scores_per_classes,
                              precisions_per_classes, recalls_per_classes)

        # assert that result file exists
        path = str(Paths.get_project_root()) + "\\Results\\" + agent_name + ".json"
        self.assertTrue(os.path.exists(path))

        # assert that plot works
        ML = {"learning_scenarios": ["ML"], "title": "ML"}
        acc = {"metrics": ["test_acc"], "xlabel": "global epochs",
               "ylabel": "accuracy score", "title": "total accuracy scores", "kind": "line"}
        mock_plt = mock.MagicMock()
        Metrics.plot_metrics(args, "test", "test", ML, acc, mock_plt)
        mock_plt.show.assert_called_once()

    def test_average_weights(self):
        # set variables
        avg = {}
        cnn1 = CNN(10)
        cnn2 = CNN(10)
        weights = {"1": cnn1.state_dict(), "2": cnn2.state_dict()}

        # average gradients
        Learning.average_weights(avg, weights)

        # assert that calculated correctly
        for i in avg["weights"].keys():
            list_avg = avg["weights"][i].tolist()
            list_g1 = weights["1"][i].tolist()
            list_g2 = weights["2"][i].tolist()
            self.control(list_avg, list_g1, list_g2)

        # test that the gradients are loadable in a new optimizer
        cnn_avg = CNN(10)
        cnn_avg.load_state_dict(avg["weights"])

    def test_average_gradients(self):
        # set variables
        args = Argparser.args_parser()
        avg = {}
        batch_sizes = {'3': 10, '9': 10, '2': 20}
        cnn1 = CNN(10)
        cnn2 = CNN(10)
        cnn3 = CNN(10)
        criterion = nn.MSELoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        gradients = {}
        opt1 = torch.optim.SGD(cnn1.parameters(), lr=0.1, momentum=0.9)
        opt2 = torch.optim.SGD(cnn2.parameters(), lr=0.1, momentum=0.9)
        training_losses1 = []
        training_losses2 = []
        x_train, y_train, y_train_original_labels = Data.prepare_dataset(
            args.dataset_training, batch_sizes, local_epochs=10)

        # train both models so that gradients are calculated
        Learning.training(criterion, device, cnn1, opt1, training_losses1, x_train, y_train)
        Learning.training(criterion, device, cnn2, opt2, training_losses2, x_train, y_train)

        # average gradients
        gradients["1"] = opt1.state_dict()
        gradients["2"] = opt2.state_dict()
        Learning.average_gradients(avg, gradients)

        # assert that calculated correctly
        for i in avg["gradients"]["state"]:
            for j in avg["gradients"]["state"][i].keys():
                list_avg = avg["gradients"]["state"][i][j].tolist()
                list_g1 = gradients["1"]["state"][i][j].tolist()
                list_g2 = gradients["2"]["state"][i][j].tolist()
                self.control(list_avg, list_g1, list_g2)

        # assert that the gradients are loadable in a new optimizer
        opt_avg = torch.optim.SGD(cnn3.parameters(), lr=0.1, momentum=0.9)
        before = opt_avg.state_dict()["state"]
        opt_avg.load_state_dict(avg["gradients"])
        after = opt_avg.state_dict()["state"]
        self.assertNotEqual(after, before)


if __name__ == "__main__":
    unittest.main()
