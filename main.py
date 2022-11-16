from __future__ import unicode_literals, print_function, division

import torch

import os
from multiprocessing import cpu_count
import train


if __name__ == '__main__':
    train.train()
