from __future__ import unicode_literals, print_function, division

import torch

import os
from multiprocessing import cpu_count
import train


def pytorch_use_all_cpu():
    """
    设置pytorch使用全部cpu核心进行训练
    :return:
    """
    cpu_num = cpu_count()  # 自动获取最大核心数目
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

if __name__ == '__main__':
    train.train()
