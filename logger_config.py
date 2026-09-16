import logging
import yaml
import os


with open(
    "config/config.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)


log_file = config["logging"]["file"]


os.makedirs(
    os.path.dirname(log_file),
    exist_ok=True
)


logger = logging.getLogger("SeismicAI")

logger.setLevel(logging.INFO)


handler = logging.FileHandler(
    log_file,
    encoding="utf-8"
)


formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)


handler.setFormatter(
    formatter
)


if not logger.handlers:
    logger.addHandler(handler)