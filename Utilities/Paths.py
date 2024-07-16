from pathlib import Path
import yaml


class Paths():
    # get the root of the project
    @staticmethod
    def get_project_root() -> Path:
        return Path(__file__).parent.parent

# read config file
with open(str(Paths.get_project_root()) + "\\Configuration\\config.yml", "rt") as config_file:
    config = yaml.safe_load(config_file)
