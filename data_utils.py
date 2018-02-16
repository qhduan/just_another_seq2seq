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


def batch_flow(data, ws, batch_size):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    """

    all_data = list(zip(*data))

    if isinstance(ws, list) or isinstance(ws, tuple):
        assert len(ws) == len(data), \
            'len(ws) must equal to len(data) if ws is list or tuple'

    while True:
        data_batch = random.sample(all_data, batch_size)
        batches = [[] for i in range(len(data) * 2)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([len(x[j]) for x in data_batch])
            max_lens.append(max_len)

        for i, d in enumerate(data_batch):
            for j in range(len(data)):
                if isinstance(ws, list) or isinstance(ws, tuple):
                    w = ws[j]
                else:
                    w = ws
                x, xl = transform_sentence(d[j], w, max_lens[j])
                batches[j*2].append(x)
                batches[j*2+1].append(xl)
        batches = [np.asarray(x) for x in batches]

        yield batches


def test_batch_flow():
    """test batch_flow function"""
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow([x_data, y_data], [ws_input, ws_target], 4)
    x, xl, y, yl = next(flow)
    print(x.shape, y.shape, xl.shape, yl.shape)


if __name__ == '__main__':
    test_batch_flow()
