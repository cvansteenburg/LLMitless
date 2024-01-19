import logging.config
from pathlib import Path

import tomli


def init_logging(config_file: str):
    with open(config_file, "rb") as toml_file:
        try:
            config_data = tomli.load(toml_file)
        except tomli.TOMLDecodeError as e:
            print(f"Error decoding {config_file}: {e}")

    logging_config = config_data.get("tool", {}).get("LLMitless", {}).get("logging")

    if logging_config:
        for handler_config in logging_config.get("handlers", {}).values():
            if 'filename' in handler_config:
                log_file = Path(handler_config['filename'])
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.touch(exist_ok=True)

        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.log(logging.WARNING, f"No logging configuration found in {config_file}")
        print(f"No logging configuration found in {config_file}")

    logger = logging.getLogger("LLMitless") #keep as package name for config propogation
    return logger