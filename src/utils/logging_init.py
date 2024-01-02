import logging.config

import tomli


def init_logging(config_file: str):
    with open(config_file, "rb") as toml_file:
        try:
            config_data = tomli.load(toml_file)
        except tomli.TOMLDecodeError as e:
            print(f"Error decoding {config_file}: {e}")

    logging_config = config_data.get("tool", {}).get("llm_testbed", {}).get("logging")

    if logging_config:
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.log(logging.WARNING, f"No logging configuration found in {config_file}")
        print(f"No logging configuration found in {config_file}")

    logger = logging.getLogger("llm_testbed") #keep as package name for config propogation
    return logger