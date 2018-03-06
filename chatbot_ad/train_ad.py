"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
# import jieba

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
    from threadedgenerator import ThreadedGenerator

    x_data, y_data, ws = pickle.load(
        open('chatbot.pkl', 'rb'))

    vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

    # 训练部分
    n_epoch = 5
    batch_size = 64
    # x_data, y_data = shuffle(x_data, y_data, random_state=0)
    # x_data = x_data[:500000]
    # y_data = y_data[:500000]
    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './s2ss_chatbot_ad.ckpt'
    forward_path = './s2ss_chatbot_forward.ckpt'
    discriminative_path = './s2ss_chatbot_discriminative.ckpt'

    # 读取反向模型 seq2seq(x|y)
    with tf.device('/cpu:0'):
        graph_d = tf.Graph()
        with graph_d.as_default():
            random.seed(0)
            np.random.seed(0)
            tf.set_random_seed(0)

            sess_d = tf.Session(config=config)

            model_d = Discriminative(
                input_vocab_size=len(ws),
                batch_size=batch_size,
                bidirectional=bidirectional,
                cell_type=cell_type,
                depth=depth,
                use_residual=use_residual,
                use_dropout=use_dropout,
                parallel_iterations=32,
                time_major=time_major,
                hidden_units=64,
                learning_rate=0.001,
                optimizer='adam',
                dropout=0.4
            )
            init = tf.global_variables_initializer()
            sess_d.run(init)
            model_d.load(sess_d, discriminative_path)

    # 构建要训练的模型
    graph_ad = tf.Graph()
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
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            hidden_units=hidden_units,
            learning_rate=0.001,
            optimizer='adam',
            dropout=0.4,
            time_major=time_major,
            share_embedding=True
        )

        init = tf.global_variables_initializer()
        sess_ad.run(init)
        if preload:
            model_ad.load(sess_ad, forward_path)

    # 开始训练
    flow = batch_flow([x_data, y_data], ws, batch_size, raw=True)

    def flow_data(flow):
        """包一层"""
        for x, xl, xraw, y, yl, yraw in flow:

            rewards = model_d.predict(sess_d, x, xl, y, yl)
            rewards = rewards[:, 1]

            texts = []
            for i in range(batch_size):
                # text = ws.inverse_transform(y[i])
                # text = ''.join(text)[:yl[i]]
                text = ''.join(yraw[i])
                texts.append(text)
            tfidfs = np.sum(vectorizer.transform(texts), axis=1)
            tfidfs_sum = np.sum(tfidfs)

            def smooth(x):
                """把数据平整到0.5~1.5左右"""
                return (0.5 + x) * (2.0/3)

            for i in range(batch_size):
                text = texts[i]
                base_rewards = rewards[i] # smooth(rewards[i])
                repeat_rewards = repeat_reward(text) # smooth(repeat_reward(text))
                chinese_rewards = chinese_reward(text) # smooth(chinese_reward(text))
                similarity_rewards = similarity_reward(''.join(xraw[i]), text)
                # smooth(similarity_reward(''.join(xraw[i]), text))
                tfidf_rewards = tfidfs[i] / tfidfs_sum * batch_size
                # tfidf_rewards = smooth(
                #     tfidfs[i] / tfidfs_sum * batch_size)[0][0]

                # print(''.join(xraw[i]))
                # print(text)
                # print(
                #     base_rewards, repeat_rewards,
                #     chinese_rewards, similarity_rewards,
                #     tfidf_rewards
                # )

                rewards[i] = base_rewards
                rewards[i] *= repeat_rewards
                # rewards[i] *= chinese_rewards
                rewards[i] *= similarity_rewards
                rewards[i] *= tfidf_rewards

                # print(rewards[i])
                # print('-' * 30)
            # exit(1)

            rewards = rewards.reshape(-1, 1)
            yield x, xl, y, yl, rewards

    new_flow = ThreadedGenerator(flow_data(flow), queue_maxsize=10)

    for epoch in range(1, n_epoch + 1):
        costs = []
        bar = tqdm(range(steps), total=steps,
                   desc='epoch {}, loss=0.000000'.format(epoch))
        for _ in bar:

            x, xl, y, yl, rewards = next(new_flow)

            cost = model_ad.train(sess_ad, x, xl, y, yl, rewards)

            costs.append(cost)
            # lengths.append(np.mean(al))
            des = ('epoch {} '
                   'loss={:.6f} '
                   'rmean={:.4f} '
                   'rmin={:.4f} '
                   'rmax={:.4f} '
                   'rmed={:.4f}')
            bar.set_description(des.format(
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
    from collections import Counter
    arr = list(arr)
    counter = Counter(arr)
    t = sum([i for i in counter.values() if i > 1])
    return max(0, 1 - t / len(counter))


def chinese_reward(text):
    """中文越多分数越高
    chiese_reward("⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎")
    """
    import re
    return len(re.findall(r'[\u4e00-\u9fff，。！？]', text)) / len(text)


def similarity_reward(q, a):
    """越相似，reward越小"""
    from nltk.metrics.distance import edit_distance
    return edit_distance(q, a) / max(len(q), len(a))


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(False, 'lstm', 1, 'Bahdanau', False, False, False, 1024)


if __name__ == '__main__':
    main()
