"""Customized logging for tomo-center.

Adapted from tomocupy's logging module (BSD-3, UChicago Argonne LLC) — same
formatter and setup helper, scoped to the `tomo_center.*` logger tree.
"""
import logging
import traceback
from logging import *  # noqa: F401,F403  (re-export std logging API)

__all__ = ["setup_custom_logger", "ColoredLogFormatter", "log_exception"] + logging.__all__


def log_exception(logger, err, fmt="%s"):
    tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
    tb_lines = [ln for lns in tb_lines for ln in lns.splitlines()]
    for tb_line in tb_lines:
        logger.error("      %s", tb_line)


def setup_custom_logger(lfname: str = None, stream_to_console: bool = True, level=logging.INFO):
    """Attach console (colored) and optional file handlers to the package logger."""
    parent_name = __name__.split(".")[0]  # "tomo_center"
    parent_logger = logging.getLogger(parent_name)
    parent_logger.setLevel(logging.DEBUG)
    if lfname is not None:
        fh = logging.FileHandler(lfname)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s(%(lineno)s) - %(levelname)s: %(message)s"))
        fh.setLevel(logging.DEBUG)
        parent_logger.addHandler(fh)
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredLogFormatter("%(asctime)s - %(message)s"))
        ch.setLevel(level)
        parent_logger.addHandler(ch)


class ColoredLogFormatter(logging.Formatter):
    """Console formatter that colors messages by level."""
    __BLUE = "\033[94m"
    __GREEN = "\033[92m"
    __RED = "\033[91m"
    __RED_BG = "\033[41m"
    __YELLOW = "\033[33m"
    __ENDC = "\033[0m"

    def _format_message_level(self, message, level):
        colors = {
            "INFO": self.__GREEN,
            "WARNING": self.__YELLOW,
            "ERROR": self.__RED,
            "CRITICAL": self.__RED_BG,
        }
        if level in colors:
            message = f"{colors[level]}{message}{self.__ENDC}"
        return message

    def formatMessage(self, record):
        record.message = self._format_message_level(record.message, record.levelname)
        return super().formatMessage(record)
