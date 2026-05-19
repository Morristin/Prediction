import logging
from typing import DefaultDict

import torch

from tools import load_data
from train import Model, TorchTrainer

logging.getLogger(__name__)


SLIDING_WINDOW_SIZE = TorchTrainer.SLIDING_WINDOW_SIZE


def predict(data_source: str, steps: int, /, sliding_window_size=SLIDING_WINDOW_SIZE) -> dict[str, list[float]]:
    """
    Predict prices of good in next a few days.

    :param data_source: The name of CSV file which stores previous prices of good.
    :param steps: The number of predictions.
    :param sliding_window_size: The size of sliding window.
    :return: A dict, which key is the name of good and value is the predictions.
    """

    dataset = load_data(data_source, sliding_window_size)
    predictions: dict[str, list[float]] = DefaultDict(lambda: list())

    # Setting up model
    model = Model()
    model.load_state('model.pth')
    model.model.eval()

    # Make predictions
    for name, data in dataset:
        with torch.no_grad():
            for _ in range(steps):
                X = torch.tensor(data.sample, dtype=torch.float32).reshape(1, len(data), 1).to(model.device)
                prediction = model.model(X).items()
                predictions[name].append(prediction * (data.max_value - data.min_value) + data.min_value)
                data.sample = data.sample[1:] + prediction

    return predictions
