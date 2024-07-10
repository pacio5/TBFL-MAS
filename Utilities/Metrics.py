import json
from matplotlib import pyplot as plt
import os
import pandas as pd
from Utilities.Paths import Paths
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Metrics():
    @staticmethod
    def calculate_f1_score_per_classes(all_labels, all_predictions, epoch, f1_scores_per_classes):
        f1_scores = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        for i in range(len(f1_scores)):
            f1_scores_per_classes[str(i)][str(epoch)] = f1_scores[i]

    @staticmethod
    def calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                          all_test_precisions, all_test_recalls, epoch):
        all_test_accuracies[str(epoch)] = accuracy_score(all_labels, all_predictions)
        all_test_f1_scores[str(epoch)] = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
        all_test_precisions[str(epoch)] = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
        all_test_recalls[str(epoch)] = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)

    @staticmethod
    def calculate_precisions_per_classes(all_labels, all_predictions, epoch, precisions_per_classes):
        precision_scores = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        for i in range(len(precision_scores)):
            precisions_per_classes[str(i)][str(epoch)] = precision_scores[i]

    @staticmethod
    def calculate_recalls_per_classes(all_labels, all_predictions, epoch, recalls_per_classes):
        recall_scores = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        for i in range(len(recall_scores)):
            recalls_per_classes[str(i)][str(epoch)] = recall_scores[i]

    @staticmethod
    def plot_metrics(filter_files, filter_agents, filter_metrics, plt=plt):
        metrics = {}
        for filename in os.listdir(str(Paths.get_project_root()) + "\\Results"):
            if filter_files in filename:
                path = str(Paths.get_project_root()) + "\\Results\\" + filename
                if os.path.getsize(path) > 0:
                    with open(path, "r") as metrics_file:
                        metrics[filename] = json.load(metrics_file)
                else:
                    return

        filtered = {}
        for key, value in metrics.items():
            if filter_agents in key:
                for key2 in value.keys():
                    if filter_metrics in key2:
                        filtered[key + ", " + key2] = value[key2]

        df = pd.DataFrame.from_dict(filtered, orient='columns')
        df.plot(title=" performance metrics", xlabel="global epochs", ylabel="percentage", figsize=(10, 6))
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize="x-small")
        plt.subplots_adjust(bottom=0.25)
        plt.show()

    @staticmethod
    def store_metrics(agent_name, all_test_accuracies, all_test_f1_scores, all_test_precisions, all_test_recalls,
                      all_testing_losses, all_training_losses, args, batch_sizes_per_classes,
                      f1_scores_per_classes, precisions_per_classes, recalls_per_classes):
        path = str(Paths.get_project_root()) + "\\Results\\" + agent_name + ".json"
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, "r") as metrics_file:
                metrics = json.load(metrics_file)
        else:
            metrics = {"batch_sizes_per_classes": batch_sizes_per_classes}
        key = args.algorithm
        if args.iid == 0:
            key += ", non-IID"
        if args.new_entry_or_leave == "new entry":
            key += ", new entry"
        elif args.new_entry_or_leave == "leave":
            key += ", leave"
        if args.standard_deviation_for_noises == 1:
            key += ", low noises"
        if args.standard_deviation_for_noises == 2:
            key += ", middle noises"
        if args.standard_deviation_for_noises == 3:
            key += ", high noises"
        if args.wait_until_threshold_is_reached != "no threshold":
            key += ", " + str(args.wait_until_threshold_is_reached) + "=" + str(args.threshold)

        metrics[key + ", test_acc"] = all_test_accuracies
        metrics[key + ", test_f1"] = all_test_f1_scores
        metrics[key + ", test_pre"] = all_test_precisions
        metrics[key + ", test_rec"] = all_test_recalls
        metrics[key + ", test_loss"] = all_testing_losses
        metrics[key + ", train_loss"] = all_training_losses
        for i in range(args.number_of_classes_in_dataset):
            metrics[key + ", f1_cla" + str(i)] = f1_scores_per_classes[str(i)]
            metrics[key + ", pre_cla" + str(i)] = precisions_per_classes[str(i)]
            metrics[key + ", rec_cla" + str(i)] = recalls_per_classes[str(i)]

        with open(path, "w") as metrics_file:
            json.dump(metrics, metrics_file)
