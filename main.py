from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train.train()
