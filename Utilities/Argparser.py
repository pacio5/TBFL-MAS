import argparse
from Utilities.Paths import config, Paths
import yaml

class Argparser():
    @staticmethod
    def args_parser():
        path_to_launch_config = str(Paths.get_project_root()) + "\\Configuration\\launch_config.yml"
        launch_conf = yaml.load(open(path_to_launch_config), Loader=yaml.FullLoader)
        parser = argparse.ArgumentParser()

        # add parsers
        parser.add_argument('--algorithm', type=str, choices=config["options"]["algorithm"],
                            default=config["default"]["algorithm"],
                            help="algorithm that is used for learning, default is FedAvg")

        parser.add_argument('--batch_size_testing', type=int,
                            choices=config["options"]["batch_size_total"],
                            default=config["default"]["batch_size_testing"],
                            help="batch size that is used for testing")

        parser.add_argument('--batch_size_training', type=int,
                            choices=config["options"]["batch_size_total"],
                            default=config["default"]["batch_size_training"],
                            help="batch size that is used for training")

        parser.add_argument('--dataset_testing', type=str, default=config["default"]["dataset_testing"],
                            help="path to testing data set")

        parser.add_argument('--dataset_training', type=str, default=config["default"]["dataset_training"],
                            help="path to testing data set")

        parser.add_argument('--epoch', type=int, default=1,
                            help="current epoch")

        parser.add_argument('--global_epochs', type=int, default=config["default"]["global_epochs"],
                            help="number of global epochs")

        parser.add_argument('--iid', type=int, choices=config["options"]["iid"], default=config["default"]["iid"],
                            help="default is IID, change the value to 0 for non-IID")

        parser.add_argument('--jid_server', type=str, default=config["default"]["jid_server"])

        parser.add_argument('--launch_config', type=list,
                            default=list(launch_conf.values()),
                            help="launch configuration")

        parser.add_argument('--learning_rate', type=float, default=config["default"]["learning_rate"],
                            help='learning rate')

        parser.add_argument('--local_epochs', type=int, default=config["default"]["local_epochs"],
                            help="the number of local epochs")

        parser.add_argument('--new_entry_or_leave', type=str,
                            choices=config["options"]["new_entry_or_leave"],
                            default=config["default"]["new_entry_or_leave"],
                            help="settings for new entry of an agent or agents leaves the MAS,  default is no entry or leave")

        parser.add_argument('--number_of_client_agents', type=int, default=config["default"]["number_of_client_agents"],
                            help="number of client agents")

        parser.add_argument('--number_of_classes_in_dataset', type=int,
                            default=config["default"]["number_of_classes_in_dataset"],
                            help="number of classes in dataset")

        parser.add_argument('--standard_deviation_for_noises', type=float,
                            default=config["default"]["standard_deviation_for_noises"],
                            help="standard deviation for noises")

        parser.add_argument('--wait_until_threshold_is_reached', type=str,
                            choices=config["options"]["wait_until_threshold_is_reached"],
                            default=config["default"]["wait_until_threshold_is_reached"],
                            help="settings for waiting until a certain threshold is reached, default is no threshold")

        parser.add_argument('--threshold', type=float, default=config["default"]["threshold"],
                            help="threshold needed to reach before learning ends")

        args = parser.parse_args()

        return args
