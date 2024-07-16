import json

import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from Utilities.Paths import Paths, config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Metrics:
    # calculate f1 scores for every class in the dataset
    @staticmethod
    def calculate_f1_score_per_classes(all_labels, all_predictions, epoch, f1_scores_per_classes):
        f1_scores = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        for i in range(len(f1_scores)):
            f1_scores_per_classes[str(i)][str(epoch)] = f1_scores[i]

    # calculate accuracy, f1, precision and recall scores
    @staticmethod
    def calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                          all_test_precisions, all_test_recalls, epoch):
        all_test_accuracies[str(epoch)] = accuracy_score(all_labels, all_predictions)
        # calculate in total, not in every class
        all_test_f1_scores[str(epoch)] = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
        all_test_precisions[str(epoch)] = precision_score(all_labels, all_predictions, average="weighted",
                                                          zero_division=0)
        all_test_recalls[str(epoch)] = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)

    # calculate precision scores for every class in the dataset
    @staticmethod
    def calculate_precisions_per_classes(all_labels, all_predictions, epoch, precisions_per_classes):
        precision_scores = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        for i in range(len(precision_scores)):
            precisions_per_classes[str(i)][str(epoch)] = precision_scores[i]

    # calculate recall scores for every class in the dataset
    @staticmethod
    def calculate_recalls_per_classes(all_labels, all_predictions, epoch, recalls_per_classes):
        recall_scores = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        for i in range(len(recall_scores)):
            recalls_per_classes[str(i)][str(epoch)] = recall_scores[i]

    # plot the metrics
    @staticmethod
    def plot_metrics(args, filter_files, filter_agents, filter_learning_scenarios, filter_metrics, plt=plt):
        colors_styles = ['b', 'y', 'r', 'c', 'g', 'm', 'k', 'brown', 'grey', 'violet', 'pink', 'indigo', 'olive']
        marker_styles = ['v', '^', '<', '>', '8', 's', 'P', 'h', '*', 'o', 'X', 'D', 'd']
        line_styles = ['-', '-.', ':', '--']
        colors_per_agent = {}
        metrics = {}
        style = []
        colors = []
        x = 0
        for filename in os.listdir(str(Paths.get_project_root()) + "\\Results"):
            if filter_files in filename:
                path = str(Paths.get_project_root()) + "\\Results\\" + filename
                if os.path.getsize(path) > 0:
                    with open(path, "r") as metrics_file:
                        metrics[filename] = json.load(metrics_file)
                        colors_per_agent[filename] = colors_styles[x]
                        if len(colors_styles)-1 > x:
                            x += 1
                        else:
                            x = 0
                else:
                    return

        list_keys = list(metrics.keys())
        count = 0
        for item in list_keys:
            if filter_agents in item:
                count += 1

        filtered = {}
        for key, value in metrics.items():
            if filter_agents in key:
                for key2 in value.keys():
                    if key2 == filter_metrics["metrics"][0]:
                        filtered[key2] = value[key2]
                        colors = colors_per_agent[key]
                    else:
                        array = key2.split(",")
                        metric = array[-1][1:]
                        learning_scenario = ""
                        for line in range(len(array) - 1):
                            learning_scenario += array[line]
                        for i in range(len(filter_learning_scenarios["learning_scenarios"])):
                            for j in range(len(filter_metrics["metrics"])):
                                if (filter_metrics["metrics"][j] == metric and
                                        filter_learning_scenarios["learning_scenarios"][i] == learning_scenario):
                                    filtered[key + ", " + key2] = value[key2]
                                    if len(filter_metrics["metrics"]) < len(filter_learning_scenarios["learning_scenarios"]) \
                                            and count == 1:
                                        style.append(marker_styles[i] + line_styles[j])
                                        colors.append(colors_styles[i])
                                    elif count == 1:
                                        style.append(marker_styles[j] + line_styles[i])
                                        colors.append(colors_styles[j])
                                    elif count > 1:
                                        style.append(marker_styles[i] + line_styles[i])
                                        colors.append(colors_per_agent[key])


        df = pd.DataFrame.from_dict(filtered, orient='columns')
        df.plot(title=filter_learning_scenarios["title"] + filter_metrics["title"],
                xlabel=filter_metrics["xlabel"], ylabel=filter_metrics["ylabel"], figsize=(10, 6),
                kind=filter_metrics["kind"], color=colors, style=style)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3,
                   fontsize="x-small")
        plt.subplots_adjust(bottom=0.3)
        plt.show()

    # store the metrics
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
        if args.global_epochs != config["default"]["global_epochs"]:
            key += ", global epochs=" + str(args.global_epochs)
        if args.iid == 0:
            key += ", non-IID"
        if args.new_entry_or_leave == "new entry":
            key += ", new entry"
        elif args.new_entry_or_leave == "leave":
            key += ", leave"
        if args.number_of_client_agents != config["default"]["number_of_client_agents"]:
            key += ", clients=" + str(args.number_of_client_agents)
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
