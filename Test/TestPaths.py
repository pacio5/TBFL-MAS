import os
from pathlib import Path
from Utilities.Paths import Paths
import unittest


class TestPaths(unittest.TestCase):
    def test_paths(self):
        # set variable
        project_root = Path(__file__).parent.parent

        # assert if project root is correctly given by the function from different directories
        self.assertEqual(project_root, Paths.get_project_root())
        os.chdir("..")
        self.assertEqual(project_root, Paths.get_project_root())
        os.chdir("Behaviours")
        self.assertEqual(project_root, Paths.get_project_root())


if __name__ == "__main__":
    unittest.main()
