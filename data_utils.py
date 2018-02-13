"""
一些数据操作所需的模块
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

VOCAB_SIZE_THRESHOLD_CPU = 50000


def _get_available_gpus():
    """获取当前可用GPU数量"""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size.
    根据输入输出的字典大小，选择在CPU还是GPU上初始化embedding向量
    """
    gpus = _get_available_gpus()
    if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    return "/gpu:0"



def transform_data(q, a, ws_q, ws_a, q_max, a_max):
    """转换数据
    """
    x = ws_q.transform(q, max_len=q_max)#, add_end=True)
    y = ws_a.transform(a, max_len=a_max)
    xl = len(q) # q_max# + 1 # len(q)
    yl = len(a) # a_max # len(a)
    # yl = max_len
    return x, xl, y, yl


def batch_flow(x_data, y_data, ws_q, ws_a, batch_size):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    """

    all_data = list(zip(x_data, y_data))

    while True:

        data_batch = random.sample(all_data, batch_size)

        q_max = max([len(x[0]) for x in data_batch])
        a_max = max([len(x[1]) for x in data_batch])
        data_batch = sorted(data_batch, key=lambda x: len(x[1]), reverse=True)

        x_batch = []
        y_batch = []
        xlen_batch = []
        ylen_batch = []

        for q, a in data_batch:
            x, xl, y, yl = transform_data(
                q, a, ws_q, ws_a, q_max, a_max
            )
            x_batch.append(x)
            xlen_batch.append(xl)
            y_batch.append(y)
            ylen_batch.append(yl)

        yield (
            np.array(x_batch),
            np.array(xlen_batch),
            np.array(y_batch),
            np.array(ylen_batch)
        )



def batch_flow_bucket(x_data, y_data, ws_q, ws_a, batch_size, n_bucket=4):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    一个 trick
    相当于把不同数据的根据 target 句子的长度分组，算是一种 bucket
    这里弄的比较简单，复杂一点的是把“相近长度”的输出聚合到一起
    例如输出句子长度1~3的一组，4~6的一组
    每个batch不会出现不同组的长度
    """
    sizes = sorted(list(set([len(y) for y in y_data])))
    buckets = (np.linspace(0, 1, n_bucket + 1) * len(sizes)).astype(int)
    print('buckets', buckets)

    sizes_data = {}
    for i, k in enumerate(buckets):
        if i > 0:
            low = buckets[i - 1]
            v = [(x, y) for x, y in zip(x_data, y_data)
                 if len(y) > low and len(y) < k]
            sizes_data[k] = v
            while len(sizes_data[k]) < batch_size:
                sizes_data[k] = sizes_data[k] + sizes_data[k]
    sizes = sorted(list(sizes_data.keys()))
    # print('sizes', buckets)

    assert tuple(buckets[1:]) == tuple(sizes), \
        '{} != {}'.format(buckets, sizes)
    assert len(sizes) == n_bucket

    while True:

        size = random.choice(sizes)
        data_batch = random.sample(sizes_data[size], batch_size)

        q_max = max([len(x[0]) for x in data_batch])
        a_max = max([len(x[1]) for x in data_batch])
        data_batch = sorted(data_batch, key=lambda x: len(x[1]), reverse=True)

        x_batch = []
        y_batch = []
        xlen_batch = []
        ylen_batch = []

        for q, a in data_batch:
            x, xl, y, yl = transform_data(
                q, a, ws_q, ws_a, q_max, a_max
            )
            x_batch.append(x)
            xlen_batch.append(xl)
            y_batch.append(y)
            ylen_batch.append(yl)

        yield (
            np.array(x_batch),
            np.array(xlen_batch),
            np.array(y_batch),
            np.array(ylen_batch)
        )
