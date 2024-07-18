from Utilities.Argparser import Argparser
from Utilities.Paths import Paths
import unittest
import yaml


class TestParser(unittest.TestCase):
    def test_parser(self):
        # set variable
        args = Argparser.args_parser()
        path_to_launch_config = str(Paths.get_project_root()) + "\\Configuration\\launch_config.yml"
        launch_conf = yaml.load(open(path_to_launch_config), Loader=yaml.FullLoader)

        # assert default values
        self.assertEqual("ML", args.algorithm)
        self.assertEqual(300, args.batch_size_training)
        self.assertEqual(300, args.batch_size_testing)
        self.assertEqual("\\fashion dataset\\fashion-mnist_test.csv", args.dataset_testing)
        self.assertEqual("\\fashion dataset\\fashion-mnist_train.csv", args.dataset_training)
        self.assertEqual(1, args.epoch)
        self.assertEqual(3, args.global_epochs)
        self.assertEqual(1, args.iid)
        self.assertEqual("server@localhost", args.jid_server)
        self.assertEqual(list(launch_conf.values())[0], args.launch_config)
        self.assertEqual(0.01, args.learning_rate)
        self.assertEqual(5, args.local_epochs)
        self.assertEqual("no changes", args.new_entry_or_leave)
        self.assertEqual(5, args.number_of_client_agents)
        self.assertEqual(10, args.number_of_classes_in_dataset)
        self.assertEqual(0, args.standard_deviation_for_noises)
        self.assertEqual("no threshold", args.wait_until_threshold_is_reached)
        self.assertEqual(0.7, args.threshold)


if __name__ == "__main__":
    unittest.main()
