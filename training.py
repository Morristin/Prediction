import logging

import torch
from torch.utils.data import Dataset

logging.getLogger(__name__)


class PriceDataset(Dataset):
    SLIDING_WINDOW_SIZE = 30

    def __init__(self, file: str):
        """
        Detect and calculate tensor data from CSV file.

        **This function would not handle exceptions**.
        Callers need to handle exceptions themselves or make sure source file is valid.

        :param file: A CSV file which column header contains `name`, `price` and `date`.
        """

        self.training_data: list[tuple[list[float], float]] = list()
        table = open(file, 'r').readlines()

        # Process the column header of CSV file.
        column_header = tuple(unit.strip().lower() for unit in table[0].split(','))
        data_index = {detect_part: column_header.index(detect_part) for detect_part in ('name', 'price', 'date')}

        # Convert all data using sliding window method.
        class TrainingDataSubset:
            def __init__(self):
                self.prices: list[float] = list()
                self._max_price, self._min_price = None, None

            def add(self, price: float):
                self.prices.append(price)
                self._max_price = max(price, self._max_price) if self._max_price is not None else price
                self._min_price = min(price, self._min_price) if self._min_price is not None else price

            def training_data(self, sliding_window_size: int) -> list[tuple[list[float], float]]:
                """:return: The normalization result of training data in sliding window format."""

                self.prices = [(price - self._min_price) / (self._max_price - self._min_price) for price in self.prices]
                return [
                    (self.prices[index - sliding_window_size - 1 : -1], self.prices[-1])
                    for index in range(sliding_window_size, len(self.prices))
                ]

        # Special treat the first line of data. Make sure arguments are assigned before used.
        name: str = table[1][data_index['name']]
        data_subset: TrainingDataSubset = TrainingDataSubset()
        for data in table[1:]:
            data = tuple(unit.strip() for unit in data.split(','))

            if data[data_index['name']] == name:
                # TODO: Check the date and create lack data using linear interpolation.
                data_subset.add(float(data[data_index['price']]))
            else:
                self.training_data += data_subset.training_data(sliding_window_size=self.SLIDING_WINDOW_SIZE)
                name, data_subset = data[data_index['name']], TrainingDataSubset()

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.training_data[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TorchTrainer:
    # noinspection PyUnresolvedReferences
    def __init__(self):
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is not None:
            logging.info(f'Detect accelerator {accelerator.type}. Use {accelerator.type} for training.')
            self.device: torch.device = torch.device(accelerator.type)
        else:
            logging.warning('Can not detect any accelerator, use CPU to train model.')
            self.device: torch.device = torch.device('cpu')
