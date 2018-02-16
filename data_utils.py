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


def transform_sentence(q, ws_q, q_max):
    x = ws_q.transform(q, max_len=q_max)
    xl = len(q)
    return x, xl


def transform_data(q, a, ws_q, ws_a, q_max, a_max):
    """转换数据
    """
    x, xl = transform_sentence(q, ws_q, q_max)
    y, yl = transform_sentence(a, ws_a, a_max)
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
            if len(v) >= batch_size:
                sizes_data[k] = v
            # while len(sizes_data[k]) < batch_size:
            #     sizes_data[k] = sizes_data[k] + sizes_data[k]
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


def batch_flow_bucket_rl(p1_data, q1_data, p2_data, ws, batch_size, n_bucket=4):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    一个 trick
    相当于把不同数据的根据 target 句子的长度分组，算是一种 bucket
    这里弄的比较简单，复杂一点的是把“相近长度”的输出聚合到一起
    例如输出句子长度1~3的一组，4~6的一组
    每个batch不会出现不同组的长度
    """
    from data_utils import transform_sentence
    from word_sequence import WordSequence

    sizes = sorted(list(set([len(p2) for p2 in p2_data])))
    buckets = (np.linspace(0, 1, n_bucket + 1) * len(sizes)).astype(int)
    print('buckets', buckets)

    sizes_data = {}
    for i, k in enumerate(buckets):
        if i > 0:
            low = buckets[i - 1]
            v = [(p1, q1, p2) for p1, q1, p2 in zip(p1_data, q1_data, p2_data)
                 if len(p2) > low and len(p2) < k]
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

        p1_max = max([len(x[0]) for x in data_batch])
        q1_max = max([len(x[1]) for x in data_batch])
        p1q1_max = max([len(x[0]) + len(x[1]) + 1 for x in data_batch])
        p2_max = max([len(x[2]) for x in data_batch])
        data_batch = sorted(data_batch, key=lambda x: len(x[2]), reverse=True)

        p1_batch = []
        q1_batch = []
        p1q1_batch = []
        p2_batch = []

        p1_len_batch = []
        q1_len_batch = []
        p1q1_len_batch = []
        p2_len_batch = []

        for p1, q1, p2 in data_batch:

            if p1[-1] in ('。', '，', '？', '！', '。', '…', '.', ',', '?', '!', '~'):
                p1q1 = p1 + q1
            else:
                p1q1 = p1 + ['，'] + q1

            x, xl = transform_sentence(p1, ws, p1_max)
            p1_batch.append(x)
            p1_len_batch.append(xl)

            x, xl = transform_sentence(q1, ws, q1_max)
            q1_batch.append(x)
            q1_len_batch.append(xl)

            x, xl = transform_sentence(p1q1, ws, p1q1_max)
            p1q1_batch.append(x)
            p1q1_len_batch.append(xl)

            x, xl = transform_sentence(p2, ws, p2_max)
            p2_batch.append(x)
            p2_len_batch.append(xl)

        yield (
            np.array(p1_batch),
            np.array(p1_len_batch),
            np.array(q1_batch),
            np.array(q1_len_batch),

            np.array(p1q1_batch),
            np.array(p1q1_len_batch),
            np.array(p2_batch),
            np.array(p2_len_batch),
        )
