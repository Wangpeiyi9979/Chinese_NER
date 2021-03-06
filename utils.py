import os
import re
import codecs
import numpy as np
import time
import logging


def now():
    return str(time.strftime("%Y-%m-%d %H:%M:%S"))


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)
    def clear(self):
        self.steps = 0
        self.total = 0

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def create_label_dict(path, reverse=False):
    res = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            res[line.strip().split('@')[0]] = idx
    if reverse:
        res = {int(j):i for i,j in res.items()}
    return res

def create_id2describe(path):
    res = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            res[idx] = line.strip().split('@')[1]
    return res