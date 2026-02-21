"""
Logger
"""

import sys
from typing import Optional


# ANSI color codes
class _Colors:
    GREEN = "\033[92m"
    GREEN_BOLD = "\033[1;92m"
    RED = "\033[91m"
    RED_BOLD = "\033[1;91m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


class Logger:
    """
    Lightweight logger
    green  = info / success
    red    = warning / error
    gray   = debug
    """

    def __init__(self, name: str = "", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self._prefix = f"[{name}] " if name else ""

    def _log(self, color: str, label: str, msg: str, file=sys.stdout):
        prefix = f"{_Colors.CYAN}{self._prefix}{_Colors.RESET}" if self.name else ""
        print(f"{prefix}{color}{label}{_Colors.RESET} {msg}", file=file)

    def info(self, msg: str):
        """Green — general progress info"""
        self._log(_Colors.GREEN, "INFO", msg)

    def success(self, msg: str):
        """Green bold — task completed"""
        self._log(_Colors.GREEN_BOLD, " OK ", msg)

    def warning(self, msg: str):
        """Red — something to watch out for"""
        self._log(_Colors.RED, "WARN", msg, file=sys.stderr)

    def error(self, msg: str):
        """Red bold — something failed"""
        self._log(_Colors.RED_BOLD, " ERR", msg, file=sys.stderr)

    def debug(self, msg: str):
        """Gray — only shown when verbose=True"""
        if self.verbose:
            self._log(_Colors.GRAY, " DBG", msg)

    def step(self, msg: str):
        """Yellow — sub-step within a process"""
        self._log(_Colors.YELLOW, "STEP", msg)


# Global registry so same name returns same logger
_loggers: dict = {}


def get_logger(name: str = "", verbose: bool = False) -> Logger:
    """
    Get or create a named logger
    Same name returns the same logger instance
    """
    if name not in _loggers:
        _loggers[name] = Logger(name=name, verbose=verbose)
    return _loggers[name]
