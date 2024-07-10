from Utilities.Argparser import Argparser
from Utilities.Paths import Paths
import unittest

class TestParser(unittest.TestCase):
    def test_parser(self):
        # set variable
        args = Argparser.args_parser()

        # assert default values
        self.assertEqual("ML", args.algorithm)
        self.assertEqual(300, args.batch_size_training)
        self.assertEqual(300, args.batch_size_testing)
        self.assertEqual(str(Paths.get_project_root()) + "\\Configuration\\RunConfiguration\\config_FedAvg.yml", args.config_file)
        self.assertEqual("\\fashion dataset\\fashion-mnist_test.csv", args.dataset_testing)
        self.assertEqual("\\fashion dataset\\fashion-mnist_train.csv", args.dataset_training)
        self.assertEqual(5, args.global_epochs)
        self.assertEqual(1, args.iid)
        self.assertEqual("server@localhost", args.jid_server)
        self.assertEqual(0.01, args.learning_rate)
        self.assertEqual(5, args.local_epochs)
        self.assertEqual("no changes", args.new_entry_or_leave)
        self.assertEqual(10, args.number_of_agents)
        self.assertEqual(10, args.number_of_classes_in_dataset)
        self.assertEqual(0, args.standard_deviation_for_noises)
        self.assertEqual("no threshold", args.wait_until_threshold_is_reached)
        self.assertEqual(0.7, args.threshold)


if __name__ == "__main__":
    unittest.main()
