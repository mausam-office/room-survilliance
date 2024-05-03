from ctypes.wintypes import LANGID
from pathlib import Path
import re
import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

PATHS = ROOT / "configs/paths.yaml"

REQUIRED_KEYS = ['landmarks']


def yaml_load(file, initial=False):
    """
    Load YAML data from a file.

    Args:
        file (str): File name.

    Returns:
        (dict): YAML data.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-yaml file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        content = f.read()

        # remove special characters
        if not content.isprintable():
            content = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", content)

        data =  yaml.safe_load(content) or {} # always return a dict (yaml.safe_load() may return None for empty files)
        if data and initial:
            keys = list(data.keys())
            for k in REQUIRED_KEYS:
                assert k in keys, f"{k} key must be present in file {file}"
        return data

PATHS_DICT = yaml_load(PATHS, initial=True)
POSE_LANDMARKS_NAMES = yaml_load(PATHS_DICT['landmarks'])['pose']

                    


