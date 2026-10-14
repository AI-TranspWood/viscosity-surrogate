"""Logger utilities for the viscosity surrogate model."""
import logging

from rich.logging import RichHandler


def add_console_handler(logger: logging.Logger) -> None:
    """Add a console handler to the logger.

    Args:
        logger (logging.Logger): The logger to which the console handler will be added.
    """
    # Create a RichHandler for pretty logging output
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    console_formatter = logging.Formatter(
        "{asctime} - {name} - {message}", style="{",
    )
    rich_handler.setFormatter(console_formatter)

    # Add the handler to the logger
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(rich_handler)

def add_file_handler(logger: logging.Logger, output_file: str = None) -> None:
    """Add a file handler to the logger.

    Args:
        logger (logging.Logger): The logger to which the file handler will be added.
        output_file (str): The path to the file where the log messages will be saved.
    """
    if output_file is None:
        return
    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(output_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the file handler
    file_formatter = logging.Formatter(
        "{asctime} - {levelname:>7s} - {message}", style="{"
    )
    file_handler.setFormatter(file_formatter)

    # Add the file handler to the logger
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)


def get_logger(name: str, output_file: str = None) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name (str): The name of the logger.
        output_file (str): The path to the file where the log messages will be saved.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    add_console_handler(logger)
    add_file_handler(logger, output_file)

    return logger

def set_console_level(logger: logging.Logger, level: int) -> None:
    """Set the logging level for the console handler of the logger.

    Args:
        logger (logging.Logger): The logger whose console handler's level is to be set.
        level (int): The logging level to set for the console handler.
    """
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(level)

def set_file_level(logger: logging.Logger, level: int) -> None:
    """Set the logging level for the file handler of the logger.

    Args:
        logger (logging.Logger): The logger whose file handler's level is to be set.
        level (int): The logging level to set for the file handler.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(level)

MAIN_LOGGER = get_logger("aitw_viscosity_surrogate")
