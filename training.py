import logging

import torch

logging.getLogger(__name__)


class TorchTrainer:

    # noinspection PyUnresolvedReferences
    def __init__(self):
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is not None:
            logging.info(f'Detect accelerator {accelerator.type}. Use {accelerator.type} for training.')
            self.device = torch.device(accelerator.type)
        else:
            logging.warning(f'Can not detect any accelerator, use CPU to train model.')
            self.device = torch.device('cpu')
