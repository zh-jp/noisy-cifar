import logging
import os
import datetime
from pathlib import Path
from typing import Union
from colorlog import ColoredFormatter


def get_logger(save_prefix: Union[Path, str]):
    name = datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if os.path.exists(save_prefix) is False:
        os.mkdir(save_prefix)
    file_handler = logging.FileHandler(os.path.join(save_prefix, f'{name}.log'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # define format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    colorlog_format = '%(log_color)s' + log_format
    # create formatter
    log_formatter = logging.Formatter(log_format)
    color_formatter = ColoredFormatter(colorlog_format, reset=True)
    # using formatter
    file_handler.setFormatter(log_formatter)
    console_handler.setFormatter(color_formatter)
    return logger
