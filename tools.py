import logging


def create_logger(
    name: str,
    /,
    filename: str | None = None,
    filemode: str = 'a',
    fmt: str = '%(message)s',
    datefmt: str | None = None,
    level=logging.DEBUG,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create local logger to record information in an easy way.

    :param name: The name of the logger. It is recommended to use a **unique and meaningful** name to avoid name conflict.
    :param filename: The filename used to specifies the FileHandler that will be created. **The default value `None` means using stream** instead of file.
    :param filemode: Specifies the mode to open the file, if filename is specified. Default value is `a`.
    :param fmt: Use the specified format string for the handler.
    :param datefmt: Use the specified date/time format.
    :param level: Set the root logger level to the specified level.
    :param propagate: If False, logging messages will not be passed to the handlers of ancestor loggers.
    """

    logger = logging.getLogger(name)

    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename=filename, mode=filemode)

    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.setLevel(level)
    logger.propagate = propagate

    logger.addHandler(handler)
    return logger
