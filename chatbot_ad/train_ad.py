"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import jieba

sys.path.append('..')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units,
         preload=True):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from discriminative import Discriminative
    from data_utils import batch_flow_bucket
    from word_sequence import WordSequence # pylint: disable=unused-variable

    x_data, y_data, ws = pickle.load(
        open('chatbot.pkl', 'rb'))

    # 训练部分
    n_epoch = 10
    batch_size = 256
    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './s2ss_chatbot_ad.ckpt'
    forward_path = './s2ss_chatbot_forward.ckpt'
    discriminative_path = './s2ss_chatbot_discriminative.ckpt'

    graph_d = tf.Graph()
    graph_ad = tf.Graph()

    # 读取反向模型 seq2seq(x|y)
    with graph_d.as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        sess_d = tf.Session(config=config)

        model_d = Discriminative(
            input_vocab_size=len(ws),
            batch_size=batch_size * 2,
            learning_rate=0.001,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=32,
            time_major=time_major,
            hidden_units=hidden_units,
            optimizer='adam'
        )
        init = tf.global_variables_initializer()
        sess_d.run(init)
        model_d.load(sess_d, discriminative_path)

    # 构建要训练的模型
    with graph_ad.as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        sess_ad = tf.Session(config=config)

        model_ad = SequenceToSequence(
            input_vocab_size=len(ws),
            target_vocab_size=len(ws),
            batch_size=batch_size,
            # beam_width=12,
            learning_rate=0.01,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=64,
            hidden_units=hidden_units,
            optimizer='adadelta',
            time_major=time_major
        )

        init = tf.global_variables_initializer()
        sess_ad.run(init)
        if preload:
            model_ad.load(sess_ad, forward_path)

    # 开始训练
    flow = batch_flow_bucket(
        x_data, y_data, ws, ws, batch_size
    )

    for epoch in range(1, n_epoch + 1):
        costs = []
        lengths = []
        bar = tqdm(range(steps), total=steps,
                   desc='epoch {}, loss=0.000000'.format(epoch))
        for _ in bar:

            x, xl, y, yl = next(flow)

            _, a = model_ad.entropy(
                sess_ad, x, xl, y, yl
            )

            al = []
            new_a = []
            for aa in a:
                for j in range(0, len(aa)):
                    if aa[j] == WordSequence.END:
                        break
                new_a.append(list(aa[:j]))
                if j <= 0:
                    j = 1
                al.append(j)
            max_a_len = max([len(aa) for aa in new_a])
            a = []
            for aa in new_a:
                if len(aa) < max_a_len:
                    aa += [WordSequence.END] * (max_a_len - len(aa))
                a.append(aa)
            al = np.array(al)
            a = np.array(a)

            if a.shape[1] == 0:
                continue

            rewards = model_d.predict(sess_d, x, xl, a, al)
            rewards = rewards[:, 1]
            rewards = rewards.reshape(-1, 1)

            cost = model_ad.train(sess_ad, x, xl, y, yl, rewards)

            costs.append(cost)
            lengths.append(np.mean(al))
            bar.set_description('epoch {} loss={:.6f} c={:.4f} rs={:.4f}'.format(
                epoch,
                np.mean(costs),
                np.mean(lengths),
                np.mean(rewards)
            ))

        model_ad.save(sess_ad, save_path)


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
