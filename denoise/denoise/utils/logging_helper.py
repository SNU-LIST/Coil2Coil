import sys
import os
import datetime
import logging
from colorlog import ColoredFormatter
from colored_traceback import Colorizer

root_logger = logging.getLogger("")
package_logger = logging.getLogger(__name__.split(".")[0])
logger = logging.getLogger("")


def _log_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        root_logger.error( "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    colorizer = Colorizer("default", False)
    sys.excepthook = colorizer.colorize_traceback
    colorizer.colorize_traceback(exc_type, exc_value, exc_traceback)


def setup(args = None, log_dir = None, printing_mode = True, filename = 'log.txt'):
    if printing_mode == 'no':
        if log_dir is None:
            raise('Logging_helper: log_dir is undefined')
        if log_dir != "" and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_path = os.path.join(log_dir, filename)
        file = logging.FileHandler(file_path, mode="a")

        formatter = logging.Formatter(fmt="%(asctime)s %(message)s", 
                                      datefmt="%m-%d %H:%M:%S  ")
        file.setLevel(args.loglv)
        file.setFormatter(formatter)
        root_logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(args.loglv)
    formatter = ColoredFormatter(
            "%(log_color)s%(message)s%(reset)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG':    'green',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red,bold',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    console_handle = console
    sys.excepthook = _log_exception
    package_logger.setLevel(args.loglv)
