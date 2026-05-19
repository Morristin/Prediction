import logging

import torch
from torch.utils.data import DataLoader, Dataset

logging.getLogger(__name__)


class PriceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class PricePredictNeuralNetwork(torch.nn.Module):
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.0

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            dropout=self.DROPOUT,
        )
        self.linear = torch.nn.Linear(in_features=self.HIDDEN_SIZE, out_features=1)

    def forward(self, x):
        return self.linear(self.lstm(x)[0][:, -1, :])


class TorchTrainer:
    SLIDING_WINDOW_SIZE = 30
    TRAINING_AND_TEST_SPLIT_POINT = 0.8
    EPOCHS = 5

    # noinspection PyUnresolvedReferences
    def __init__(self):
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is not None:
            logging.info(f'Detect accelerator {accelerator.type}. Use {accelerator.type} for training.')
            self.device: torch.device = torch.device(accelerator.type)
        else:
            logging.warning('Can not detect any accelerator, use CPU to train model.')
            self.device: torch.device = torch.device('cpu')

        self.training_data, self.test_data = None, None
        self.train_dataloader, self.test_dataloader = None, None

    def load_data(self, file: str, /, sliding_window_size: int = SLIDING_WINDOW_SIZE):
        """
        Detect and calculate tensor data from CSV file.

        **This function would not handle exceptions**.
        Callers need to handle exceptions themselves or make sure source file is valid.

        :param file: A CSV file which column header contains `name`, `price` and `date`.
        :param sliding_window_size: The size of sliding window when split data into units.
        """

        training_data: list[tuple[list[float], float]] = list()
        test_data: list[tuple[list[float], float]] = list()

        table = open(file, 'r').readlines()
        logging.info(f'Read table from file: {file}')

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
                """:return: The normalization result of training data in sliding window format."""
                prices = [(price - self._min_price) / (self._max_price - self._min_price) for price in self.prices]
                return [(prices[index - window_size - 1 : -1], prices[-1]) for index in range(window_size, len(prices))]

        # Special treat the first line of data. Make sure arguments are assigned before used.
        name: str = table[1][data_index['name']]
        data_subset: TrainingDataSubset = TrainingDataSubset()
        for data in table[1:]:
            data = tuple(unit.strip() for unit in data.split(','))

            if data[data_index['name']] == name:
                # TODO: Check the date and create lack data using linear interpolation.
                data_subset.add(float(data[data_index['price']]))
            else:
                # Calculate split point to split training data and test data.
                split_point: int = int(len(data_subset) * self.TRAINING_AND_TEST_SPLIT_POINT)
                normalization_data = data_subset.normalization_data(sliding_window_size)
                training_data += normalization_data[:split_point]
                test_data += normalization_data[split_point:]

                # Record logs and reinitialize the arguments for next loop.
                logging.debug(f'Finish loading the prices data of {name}.')
                name, data_subset = data[data_index['name']], TrainingDataSubset()

        self.training_data, self.test_data = PriceDataset(training_data), PriceDataset(test_data)
        logging.info('Created training and testing Dataset.')

    def load_dataset(self):
        self.train_dataloader = DataLoader(self.training_data, batch_size=32)
        self.test_dataloader = DataLoader(self.test_data, batch_size=32)
        logging.info('Created training and testing DataLoader from Dataset.')

    # noinspection PyPep8Naming
    def train(self):
        # Predefine arguments model, loss_function and optimizer.
        model = PricePredictNeuralNetwork().to(self.device)
        logging.info(f'Move neural network model into device: {self.device.type}')
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(self.EPOCHS):
            # Model train loop.
            model.train()
            for batch, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction error
                prediction = model(X)
                loss = loss_function(prediction, y.unsqueeze(1))

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Check the model’s performance to ensure it is learning.
            model.eval()
            num_batches, test_loss = len(self.test_dataloader), 0
            with torch.no_grad():
                for X, y in self.test_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    prediction = model(X)
                    test_loss += loss_function(prediction, y).item()
            test_loss /= num_batches
