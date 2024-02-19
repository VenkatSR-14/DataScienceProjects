import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

def load_env_vars(root_dir: Union[str, Path]) -> dict:
    """

    :param root_dir of the env files:
    :return:  Dictionary with environment variable
    """
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    load_dotenv(dotenv_path=root_dir / ".env.default")
    load_dotenv(dotenv_path=root_dir / ".env", override = True)
    return dict(os.environ)

def get_root_dir(default_value: str= ".") -> Path:
    """
    Get root directory of the project
    :param default_value: returns default value of .
    :return: returns path to project
    """
    return Path(os.getenv("API_KEY_ROOT_DIR", default = default_value))

API_KEY_ROOT_DIR = get_root_dir()
SETTINGS = load_env_vars(root_dir = API_KEY_ROOT_DIR)



