
"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf
# import jieba
from tqdm import tqdm
# from sklearn.utils import shuffle

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from same_person_model import SamePerson
    from data_utils import batch_flow
    from word_sequence import WordSequence # pylint: disable=unused-variable

    x1_data, x2_data, y_data, ws = pickle.load(
        open('same_person.pkl', 'rb'))

    x_data, y_data, _ = pickle.load(
        open('chatbot.pkl', 'rb'))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './s2ss_chatbot_samperson.ckpt'
    batch_size = 1

    with tf.Graph().as_default():
        model = SamePerson(
            input_vocab_size=len(ws),
            n_target=2,
            batch_size=batch_size,
            learning_rate=0.001,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=1,
            time_major=time_major,
            hidden_units=hidden_units,
            optimizer='adam'
        )
        init = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init)
        model.load(sess, save_path)

    # 开始训练
    # flow = batch_flow([x1_data, x2_data, y_data], [ws, ws, None], batch_size, raw=True)
    flow = batch_flow([x_data, y_data], ws, batch_size, raw=True)

    steps = 100
    bar = range(steps)
    for _ in bar:

        # x1, x1l, x1r, x2, x2l, x2r, y, _, _ = next(flow)
        x1, x1l, x1r, x2, x2l, x2r = next(flow)

        print(x1r, x2r)
        # print(x1, x2)
        # print(x1.shape, x2.shape, x1l.shape, x2l.shape)

        ans = model.predict(sess, x1, x1l, x2, x2l)
        print('{:.3f}'.format(ans[0][1]))#, y[0])
        print('-' * 30)



def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
