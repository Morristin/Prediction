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
        return torch.tensor(x, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(0)


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


class Model:
    def __init__(self):
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is not None:
            logging.info(f'Detect accelerator {accelerator.type}. Use {accelerator.type} for training.')
            self.device: torch.device = torch.device(accelerator.type)
        else:
            logging.warning('Can not detect any accelerator, use CPU to train model.')
            self.device: torch.device = torch.device('cpu')

        self.model = PricePredictNeuralNetwork().to(self.device)

    def load_state(self, file: str):
        self.model.load_state_dict(torch.load(file, weights_only=True))
        logging.info(f'Loaded model state dict from {file}.')

    def save_state(self, file: str = 'model.pth'):
        torch.save(self.model.state_dict(), file)
        logging.info(f'Saved model state dict to {file}.')


class TorchTrainer(Model):
    SLIDING_WINDOW_SIZE = 30
    TRAINING_AND_TEST_SPLIT_POINT = 0.8
    EPOCHS = 5

    def __init__(self):
        super().__init__()

        self.training_data, self.test_data = None, None
        self.train_dataloader, self.test_dataloader = None, None

    def create_dataset(self, data_source: str, /, sliding_window_size: int = SLIDING_WINDOW_SIZE):
        from tools import load_data

        dataset = load_data(data_source, sliding_window_size)

        training_data: list[tuple[list[float], float]] = list()
        test_data: list[tuple[list[float], float]] = list()

        for _, data in dataset.items():
            split_point: int = int(len(data) * self.TRAINING_AND_TEST_SPLIT_POINT)
            training_data += data[:split_point]
            test_data += data[split_point:]

        self.training_data, self.test_data = PriceDataset(training_data), PriceDataset(test_data)
        logging.info('Created training and testing Dataset.')

    def create_dataloader(self):
        self.train_dataloader = DataLoader(self.training_data, batch_size=32)
        self.test_dataloader = DataLoader(self.test_data, batch_size=32)
        logging.info('Created training and testing DataLoader from Dataset.')

    # noinspection PyPep8Naming
    def train(self):
        # Predefine arguments model, loss_function and optimizer.
        logging.info(f'Move neural network model into device: {self.device.type}')
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Initialize a new logger for training process visibility.
        from tools import create_logger

        train_logger = create_logger('training_process')
        OUTPUT_FREQUENCY = 1000

        # Train model over several epochs for better predictions.
        for epoch in range(self.EPOCHS):
            train_logger.info(f'Epoch {epoch + 1}: \n' + '-' * 20)

            # Model train loop.
            self.model.train()
            train_data_size = len(self.train_dataloader.dataset)
            for batch, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction error
                prediction = self.model(X)
                loss = loss_function(prediction, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Show training process
                if batch % OUTPUT_FREQUENCY == 0:
                    train_logger.info(f'Loss: {loss.item():>7f}  [{(batch + 1) * len(X):>5d}/{train_data_size:>5d}]')

            # Check the model’s performance to ensure it is learning.
            self.model.eval()
            num_batches, test_loss = len(self.test_dataloader), 0
            with torch.no_grad():
                for X, y in self.test_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    prediction = self.model(X)
                    test_loss += loss_function(prediction, y).item()
            train_logger.info(f'Test average loss: {test_loss / num_batches:>7f}.\n')

    # noinspection PyPep8Naming
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        predictions, targets = list(), list()

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                predictions.append(self.model(X))
                targets.append(y.unsqueeze(1))

        mean_absolute_error = torch.mean(torch.abs(torch.cat(predictions) - torch.cat(targets))).item()
        root_mean_absolute_error = torch.sqrt(torch.mean((torch.cat(predictions) - torch.cat(targets)) ** 2)).item()

        return mean_absolute_error, root_mean_absolute_error
