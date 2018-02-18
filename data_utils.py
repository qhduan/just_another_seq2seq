"""
一些数据操作所需的模块
"""

import random
import numpy as np
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


def transform_sentence(sentence, ws, max_len=None):
    """转换一个单独句子
    Args:
        sentence: 一句话，例如一个数组['你', '好', '吗']
        ws: 一个WordSequence对象，转换器
        max_len:
            进行padding的长度，也就是如果sentence长度小于max_len
            则padding到max_len这么长
    Ret:
        encoded:
            一个经过ws转换的数组，例如[4, 5, 6, 3]
        encoded_len: 上面的长度
    """
    encoded = ws.transform(
        sentence,
        max_len=max_len if max_len is not None else len(sentence))
    encoded_len = len(encoded)
    return encoded, encoded_len


def batch_flow(data, ws, batch_size, raw=False):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    Args:
        data:
            是一个数组，必须包含一个护着更多个同等的数据队列数组
        ws:
            可以是一个WordSequence对象，也可以是多个组成的数组
            如果是多个，那么数组数量应该与data的数据数量保持一致，即len(data) == len(ws)
        batch_size:
            批量的大小
        raw:
            是否返回原始对象，如果为True，假设结果ret，那么len(ret) == len(data) * 3
            如果为False，那么len(ret) == len(data) * 2

    例如需要输入问题与答案的队列，问题队列Q = (q_1, q_2, q_3 ... q_n)
    答案队列A = (a_1, a_2, a_3 ... a_n)，有len(Q) == len(A)
    ws是一个Q与A共用的WordSequence对象，
    那么可以有： batch_flow([Q, A], ws, batch_size=32)
    这样会返回一个generator，每次next(generator)会返回一个包含4个对象的数组，分别代表：
    next(generator) == q_i_encoded, q_i_len, a_i_encoded, a_i_len
    如果设置raw = True，则：
    next(generator) == q_i_encoded, q_i_len, q_i, a_i_encoded, a_i_len, a_i

    其中 q_i_encoded 相当于 ws.transform(q_i)

    不过经过了batch修正，把一个batch中每个结果的长度，padding到了数组内最大的句子长度
    """

    all_data = list(zip(*data))

    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), \
            'len(ws) must equal to len(data) if ws is list or tuple'

    mul = 2
    if raw:
        mul = 3

    while True:
        data_batch = random.sample(all_data, batch_size)
        batches = [[] for i in range(len(data) * mul)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([len(x[j]) for x in data_batch])
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws
                x, xl = transform_sentence(d[j], w, max_lens[j])
                batches[j * mul].append(x)
                batches[j * mul + 1].append(xl)
                if raw:
                    batches[j * mul + 2].append(d[j])
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
