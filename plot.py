import time
from Utilities.Argparser import Argparser
from Utilities.Metrics import Metrics


# plot comparisons for different learning scenarios
def plot_comparisons_of_learning_scenarios():
    args = Argparser.args_parser()

    # different metrics
    acc = {"metrics": ["test_acc"], "xlabel": "global epochs",
           "ylabel": "accuracy score", "title": "total accuracy scores", "kind": "line"}
    loss = {"metrics": ["test_loss", "train_loss"], "xlabel": "global epochs", "ylabel": "loss",
            "title": "training loss vs testing loss", "kind": "line"}
    test = {"metrics": ["test_acc", "test_f1", "test_pre"], "xlabel": "global epochs",
            "ylabel": "score", "title": "total scores", "kind": "line"}
    f1 = {"metrics": [], "xlabel": "global epochs", "ylabel": "f1-score",
          "title": "f1-scores per classes", "kind": "line"}
    f1_cla = {}
    pre_cla = {}
    rec_cla = {}
    cla = {}
    for i in range(args.number_of_classes_in_dataset):
        f1["metrics"].append("f1_cla" + str(i))
        f1_cla[str(i)] = {"metrics": ["f1_cla" + str(i)], "xlabel": "global epochs", "ylabel": "f1-score",
                          "title": "f1-scores per class " + str(i), "kind": "line"}
        pre_cla[str(i)] = {"metrics": ["pre_cla" + str(i)], "xlabel": "global epochs", "ylabel": "precision score",
                           "title": "precision scores per class " + str(i), "kind": "line"}
        rec_cla[str(i)] = {"metrics": ["rec_cla" + str(i)], "xlabel": "global epochs", "ylabel": "recall score",
                           "title": "recall scores per class " + str(i), "kind": "line"}
        cla[str(i)] = {"metrics": ["f1_cla" + str(i), "pre_cla" + str(i), "rec_cla" + str(i)],
                       "xlabel": "global epochs", "ylabel": "score", "title": "scores per class " + str(i),
                       "kind": "line"}

    # different learning scenarios
    learning_scenario_1_2_10_12 = {"learning_scenarios": ["ML", "FedAvg", "FedSGD", "FedPER"],
                                   "title": "comparison of algorithms: "}

    learning_scenario_8_9 = {"learning_scenarios": ["FedAvg non-IID", "FedAvg non-IID clients=10"],
                             "title": "comparison of different number of agents: "}

    learning_scenario_2_4_5 = {"learning_scenarios": ["FedAvg", "FedAvg leave", "FedAvg new entry"],
                               "title": "comparison of entry and leave of an agent: "}

    learning_scenario_2_6_7 = {"learning_scenarios": ['FedAvg global epochs=5', "FedAvg", 'FedAvg acc=0.7'],
                               "title": "comparison of training time and the accuracy score: "}
    learning_scenario_2_3_12_13 = {
        "learning_scenarios": ["FedAvg", "FedAvg high noises", "FedPER", "FedPER high noises"],
        "title": "comparison of algorithms of handling noises: "}
    fedavg = {"learning_scenarios": ["FedAvg non-IID"],
              "title": "FedAvg on handling non-IID data: "}
    fedsgd = {"learning_scenarios": ["FedSGD non-IID"],
              "title": "FedSGD on handling non-IID data: "}
    fedper = {"learning_scenarios": ["FedPER non-IID"],
              "title": "FedPER on handling non-IID data: "}

    filter_agents = ["server", "client0", "client1", "client2", "client3", "client4"]

    # plot metrics with different metrics and learning scenarios
    Metrics.plot_metrics(args, "", "server", learning_scenario_1_2_10_12, test)
    Metrics.plot_metrics(args, "", "server",
                         {"learning_scenarios": ["ML"],
                          "title": "ML: "}, test)
    Metrics.plot_metrics(args, "", "server",
                         {"learning_scenarios": ["FedAvg"],
                          "title": "FedAvg: "}, test)
    Metrics.plot_metrics(args, "", "server",
                         {"learning_scenarios": ["FedSGD"],
                          "title": "FedSGD: "}, test)
    Metrics.plot_metrics(args, "", "server",
                         {"learning_scenarios": ["FedPER"],
                          "title": "FedPER: "}, test)
    Metrics.plot_metrics(args, "", "",
                         {"learning_scenarios": ["FedPER"],
                          "title": "FedPER: "}, test)

    Metrics.plot_metrics(args, "", "", learning_scenario_2_4_5, acc)
    Metrics.plot_metrics(args, "", "client0", learning_scenario_2_4_5, acc)

    Metrics.plot_metrics(args, "", "server", learning_scenario_2_6_7, acc)

    Metrics.plot_metrics(args, "", "server", learning_scenario_2_3_12_13, acc)
    Metrics.plot_metrics(args, "", "server", learning_scenario_2_3_12_13, loss)

    for agent in filter_agents:
        Metrics.plot_metrics(args, "", agent, fedavg, f1)
        Metrics.plot_metrics(args, "", agent, fedsgd, f1)
        Metrics.plot_metrics(args, "", agent, fedper, f1)
        time.sleep(10)

    Metrics.plot_metrics(args, "", "", learning_scenario_8_9, f1_cla[str(2)])
    Metrics.plot_metrics(args, "", "", learning_scenario_8_9, f1_cla[str(3)])
    Metrics.plot_metrics(args, "", "", learning_scenario_8_9, f1_cla[str(6)])
    Metrics.plot_metrics(args, "", "", learning_scenario_8_9, f1_cla[str(7)])
    Metrics.plot_metrics(args, "", "", learning_scenario_8_9, f1_cla[str(9)])


# plot customizable plot
def customizable_plot():
    args = Argparser.args_parser()
    learning_scenario = {"learning_scenarios": args.learning_scenarios_to_plot,
                         "title": args.title_learning_scenario_to_plot}
    metric = {"metrics": args.metrics_to_plot, "xlabel": args.xlabel_to_plot,
              "ylabel": args.ylabel_to_plot, "title": args.title_metrics_to_plot, "kind": "line"}
    Metrics.plot_metrics(args, "", args.agents_to_plot, learning_scenario, metric)


if __name__ == "__main__":
    args = Argparser.args_parser()
    if args.plot_mode == 0:
        plot_comparisons_of_learning_scenarios()
    else:
        customizable_plot()
