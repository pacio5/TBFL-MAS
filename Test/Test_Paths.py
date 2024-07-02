import os
from pathlib import Path
from paths import get_project_root
import unittest

class TestPaths(unittest.TestCase):

    def test_paths(self):
        project_root = Path(__file__).parent.parent
        self.assertEqual(get_project_root(), project_root)
        os.chdir("..")
        self.assertEqual(get_project_root(), project_root)
        os.chdir("Behaviours")
        self.assertEqual(get_project_root(), project_root)


if __name__ == "__main__":
    unittest.main()


