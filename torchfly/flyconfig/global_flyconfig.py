import os
import copy
import logging
import logging.config
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, List

from .utils import get_config_fullpath

logger = logging.getLogger(__name__)


class Singleton(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalFlyConfig(metaclass=Singleton):
    def __init__(self, config_path: str = None):
        self.initialized = False
        self.user_config = None
        self.system_config = None
        self.old_cwd = os.getcwd()

        if config_path is not None:
            self.initialize(config_path)

    def initialize(self, config_path: str) -> OmegaConf:
        """
        Args:
            config_path: a file or dir
        Returns:
            user_config: only return the user config
        """
        if self.initialized:
            raise ValueError("FlyConfig is already initialized!")

        # Search config file
        if os.path.isdir(config_path):
            if os.path.exists(os.path.join(config_path, "config.yaml")):
                config_file = "config.yaml"
            elif os.path.exists(os.path.join(config_path, "config.yml")):
                config_file = "config.yml"
            else:
                raise ValueError("Cannot find config.yml. Please specify `config_file`")

            config_path = os.path.join(config_path, config_file)

        system_config = load_system_config()
        user_config = load_user_config(config_path)

        config = OmegaConf.merge(system_config, user_config)

        # get current working dir
        config.flyconfig.runtime.cwd = os.getcwd()

        # change working dir
        working_dir_path = config.flyconfig.run.dir
        os.makedirs(working_dir_path, exist_ok=True)
        os.chdir(working_dir_path)

        # configure logging
        logging.config.dictConfig(OmegaConf.to_container(config.flyconfig.logging))
        logger.info("FlyConfig Initialized")
        logger.info(f"Working directory is changed to {working_dir_path}")

        # clean defaults
        del config["defaults"]

        # get system config
        self.system_config = OmegaConf.create({"flyconfig": OmegaConf.to_container(config.flyconfig)})

        # get user config
        self.user_config = copy.deepcopy(config)
        del self.user_config["flyconfig"]

        # save config
        os.makedirs(self.system_config.flyconfig.output_subdir, exist_ok=True)
        _save_config(
            filepath=os.path.join(self.system_config.flyconfig.output_subdir, "flyconfig.yml"),
            config=self.system_config
        )
        _save_config(
            filepath=os.path.join(self.system_config.flyconfig.output_subdir, "config.yml"), config=self.user_config
        )

        logger.info("\n\nConfiguration:\n" + self.user_config.pretty())
        self.initialized = True

        return self.user_config

    def is_initialized(self) -> bool:
        return self.initialized

    def clear(self) -> None:
        self.initialized = False
        self.config = None


def _save_config(filepath, config):
    "Save the config file"
    with open(filepath, "w") as f:
        OmegaConf.save(config, f)


def merge_defaults(config_dir: str, config: OmegaConf, defaults: List) -> OmegaConf:
    """
    Merge the default lists and put into the config

    Args:
        config_dir: where the config is located
        config: existing Omega config
        defaults: A list of default items
    Returns:
        new_config: config after merge
    """
    for default in defaults:
        subconfig_key = list(default)[0]
        subconfig_value = default[subconfig_key]

        subconfig_fullpath = get_config_fullpath(os.path.join(config_dir, subconfig_key), subconfig_value)
        subconfig = OmegaConf.load(subconfig_fullpath)

        config = OmegaConf.merge(config, subconfig)

    return config


def load_system_config() -> OmegaConf:
    module_path = os.path.dirname(os.path.abspath(__file__))
    system_config_path = os.path.join(module_path, "config", "flyconfig.yml")
    system_config = OmegaConf.load(system_config_path)
    system_defaults = system_config["defaults"]

    sysmte_config = merge_defaults(
        config_dir=os.path.dirname(system_config_path), config=system_config, defaults=system_defaults
    )
    return sysmte_config


def load_user_config(config_path: str) -> OmegaConf:
    if not os.path.exists(config_path):
        raise ValueError(f"Cannot find {config_path}")

    user_config = OmegaConf.load(config_path)
    user_defaults = user_config["defaults"]

    user_config = merge_defaults(config_dir=os.path.dirname(config_path), config=user_config, defaults=user_defaults)
    return user_config