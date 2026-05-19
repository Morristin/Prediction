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
        from collections import defaultdict

        self.training_data = defaultdict(lambda: list())
        table = open(file, 'r').readlines()

        # Process the column header of CSV file.
        column_header = tuple(unit.strip().lower() for unit in table[0].split(','))
        data_index = dict()
        for detect_part in ('name', 'price', 'date'):
            data_index[detect_part] = column_header.index(detect_part)

        # Convert all data into the format of PyTorch.Tensor.
        # Special treat the first line of data. Make sure arguments are assigned before used.
        name: str = table[1][data_index['name']]
        x: list[float] = list()
        for data in table[1:]:
            data = tuple(unit.strip() for unit in data.split(','))

            if data[data_index['name']] == name:
                # TODO: Check the date and create lack data using linear interpolation.
                x.append(float(data[data_index['price']]))
                if len(x) > self.SLIDING_WINDOW_SIZE:
                    self.training_data[name].append((x[-self.SLIDING_WINDOW_SIZE - 1 : -1], x[-1]))
            else:
                # Reinitialize the arguments.
                name = data[data_index['name']]
                x = [float(data[data_index['price']])]


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
