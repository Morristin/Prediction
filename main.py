import logging

from train import TorchTrainer

logging.getLogger(__name__)


def main():
    log_format = '%(asctime)s : %(levelname)s : %(name)s' + '\n' + '%(message)s'
    logging.basicConfig(filename='logs.log', format=log_format, level=logging.INFO)

    trainer = TorchTrainer()


if __name__ == '__main__':
    main()
