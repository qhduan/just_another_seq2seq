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
    from data_utils import batch_flow
    from word_sequence import WordSequence # pylint: disable=unused-variable

    x_data, y_data, ws = pickle.load(
        open('chatbot.pkl', 'rb'))

    vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

    # 训练部分
    n_epoch = 10
    batch_size = 512
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
            batch_size=batch_size,
            learning_rate=0.0001,
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
            learning_rate=0.0001,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            hidden_units=hidden_units,
            optimizer='adam',
            time_major=time_major,
            share_embedding=True
        )

        init = tf.global_variables_initializer()
        sess_ad.run(init)
        if preload:
            model_ad.load(sess_ad, forward_path)

    # 开始训练
    flow = batch_flow([x_data, y_data], ws, batch_size)

    for epoch in range(1, n_epoch + 1):
        costs = []
        lengths = []
        bar = tqdm(range(steps), total=steps,
                   desc='epoch {}, loss=0.000000'.format(epoch))
        for _ in bar:

            x, xl, y, yl = next(flow)

            rewards = model_d.predict(sess_d, x, xl, y, yl)
            rewards = rewards[:, 1]

            texts = []
            for i in range(batch_size):
                text = ws.inverse_transform(y[i])
                text = ''.join(text)[:yl[i]]
                texts.append(text)
            # tfidfs = np.sum(vectorizer.transform(texts), axis=1)
            # tfidfs_sum = np.sum(tfidfs)

            for i in range(batch_size):
                text = texts[i]
                rewards[i] *= repeat_reward(text)
                rewards[i] *= chinese_reward(text)
                # rewards[i] *= tfidfs[i] / tfidfs_sum * batch_size

            rewards = rewards.reshape(-1, 1)

            cost = model_ad.train(sess_ad, x, xl, y, yl)#, rewards)

            costs.append(cost)
            # lengths.append(np.mean(al))
            bar.set_description('epoch {} loss={:.6f} rmean={:.4f} rmin={:.4f} rmax={:.4f} rmed={:.4f}'.format(
                epoch,
                np.mean(costs),
                np.mean(rewards),
                np.min(rewards),
                np.max(rewards),
                np.median(rewards)
            ))

        model_ad.save(sess_ad, save_path)


def repeat_reward(arr):
    """重复越多，分数越低"""
    arr = list(arr)
    from collections import Counter
    counter = Counter(arr)
    t = sum([i for i in counter.values() if i > 1])
    return(max(0, 1 - t / len(counter)))


def chinese_reward(text):
    """中文越多分数越高
    chiese_reward("⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎")
    """
    import re
    return len(re.findall(r'[\u4e00-\u9fff，。！？]', text)) / len(text)


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
