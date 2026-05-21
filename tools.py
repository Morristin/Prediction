import logging

logging.getLogger(__name__)


def load_data(data_source: str, sliding_window_size: int) -> dict[str, list[tuple[list[float], float]]]:
    """
    Detect and calculate tensor data from CSV file.

    **This function would not handle exceptions**.
    Callers need to handle exceptions themselves or make sure source file is valid.

    :param data_source: A CSV file which column header contains `name`, `price` and `date`.
    :param sliding_window_size: The size of sliding window when split data into units.
    """
    dataset: dict[str, list[tuple[list[float], float]]] = dict()

    # Read source data from specific CSV file.
    table = open(data_source, 'r').readlines()
    logging.info(f'Read table from file: {data_source}')

    # Process the column header of CSV file.
    column_header = tuple(unit.strip().lower() for unit in table[0].split(','))
    data_index = {detect_part: column_header.index(detect_part) for detect_part in ('name', 'price', 'date')}

    # Convert all data using sliding window method.
    class TrainingDataSubset:
        def __init__(self):
            self.prices: list[float] = list()
            self._max_price, self._min_price = None, None

        def __len__(self):
            return len(self.prices)

        def add(self, price: float):
            self.prices.append(price)
            self._max_price = max(price, self._max_price) if self._max_price is not None else price
            self._min_price = min(price, self._min_price) if self._min_price is not None else price

        def normalization_data(self, window_size: int) -> list[tuple[list[float], float]]:
            """
            The list this function returns is a custom class which directly integrated from `list`,
            which contains two special attributes: `max_value` and `min_value`.

            :param window_size: The size of sliding window.
            :return: A custom list which stores normalization result of training data in sliding window format.
            """

            class PriceList(list):
                def __init__(self, samples, max_value: float, min_value: float):
                    super().__init__(samples)
                    self.max_value, self.min_value = max_value, min_value

            if self._max_price != self._min_price:
                prices = [(price - self._min_price) / (self._max_price - self._min_price) for price in self.prices]
            else:  # Special treat: regard every data as 0.5.
                prices = [0.5 for price in self.prices]

            sample = (
                (prices[index - window_size - 1 : index - 1], prices[index])
                for index in range(window_size + 1, len(prices))
            )
            return PriceList(sample, self._max_price, self._min_price)

    # Special treat the first line of data. Make sure arguments are assigned before used.
    name: str = table[1][data_index['name']]
    data_subset: TrainingDataSubset = TrainingDataSubset()
    for data in table[1:]:
        data = tuple(unit.strip() for unit in data.split(','))
        if data[data_index['name']] == name:
            # TODO: Check the date and create lack data using linear interpolation.
            data_subset.add(float(data[data_index['price']]))
        elif len(data_subset) > sliding_window_size + 2:
            dataset[name] = data_subset.normalization_data(sliding_window_size)
            logging.debug(f'Finish loading the prices data of {name}.')
            name, data_subset = data[data_index['name']], TrainingDataSubset()
        else:
            logging.warning(f'The number of prices data of {name} is less than sliding window size.')
            name, data_subset = data[data_index['name']], TrainingDataSubset()

    return dataset


def load_split_data(
    data_source: str, sliding_window_size: int, train_and_test_split_point: float
) -> tuple[list[tuple[list[float], float]], list[tuple[list[float], float]]]:

    training_data: list[tuple[list[float], float]] = list()
    test_data: list[tuple[list[float], float]] = list()

    dataset = load_data(data_source, sliding_window_size)
    for _, data in dataset.items():
        split_point: int = int(len(data) * train_and_test_split_point)
        training_data += data[:split_point]
        test_data += data[split_point:]

    return training_data, test_data


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
