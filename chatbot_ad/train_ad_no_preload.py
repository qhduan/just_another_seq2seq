"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import random

import numpy as np
import tensorflow as tf
from train_ad import test


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256, preload=False)


if __name__ == '__main__':
    main()
