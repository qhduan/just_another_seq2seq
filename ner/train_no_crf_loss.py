
import random
import numpy as np
import tensorflow as tf
from train_crf_loss import test

def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 1, False, True, False, 64, 'tanh',
         crf_loss=False, save_path='./s2ss_nocrf.ckpt')


if __name__ == '__main__':
    main()
