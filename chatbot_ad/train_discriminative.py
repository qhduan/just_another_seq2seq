"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf
import jieba
from tqdm import tqdm
from sklearn.utils import shuffle

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from discriminative import Discriminative
    from data_utils import batch_flow_bucket
    from word_sequence import WordSequence # pylint: disable=unused-variable

    x_data, y_data, ws = pickle.load(
        open('chatbot.pkl', 'rb'))

    # 训练部分
    n_epoch = 10
    batch_size = 512
    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    # save_path = '/tmp/s2ss_chatbot.ckpt'
    save_path = './s2ss_chatbot_discriminative.ckpt'
    save_path_forward = './s2ss_chatbot_forward.ckpt'

    graph = tf.Graph()
    graph_d = tf.Graph()

    # 测试部分
    with graph.as_default():
        model_pred = SequenceToSequence(
            input_vocab_size=len(ws),
            target_vocab_size=len(ws),
            batch_size=batch_size,
            mode='train',
            beam_width=0,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=32,
            time_major=time_major,
            hidden_units=hidden_units,
            optimizer='adadelta'
        )
        init = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init)
        model_pred.load(sess, save_path_forward)

    with graph_d.as_default():
        model_d = Discriminative(
            input_vocab_size=len(ws),
            batch_size=batch_size * 2,
            learning_rate=0.001,
            bidirectional=False,
            cell_type=cell_type,
            depth=1,
            use_residual=False,
            use_dropout=False,
            parallel_iterations=32,
            time_major=time_major,
            hidden_units=hidden_units,
            optimizer='adadelta'
        )
        init = tf.global_variables_initializer()
        sess_d = tf.Session(config=config)
        sess_d.run(init)
        # model_d.load(sess, save_path_rl)


    # 开始训练
    flow = batch_flow_bucket(
        x_data, y_data, ws, ws, batch_size
    )

    for epoch in range(1, n_epoch + 1):
        costs = []
        accuracy = []
        bar = tqdm(range(steps), total=steps,
                   desc='epoch {}, loss=0.000000'.format(epoch))
        for _ in bar:

            x, xl, y, yl = next(flow)

            _, a = model_pred.entropy(
                sess, x, xl, y, yl
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

            max_len = max((a.shape[1], y.shape[1]))
            if a.shape[1] < max_len:
                a = np.concatenate((a, np.ones((batch_size, max_len - a.shape[1])) * WordSequence.END), axis=1)
            if y.shape[1] < max_len:
                y = np.concatenate((y, np.ones((batch_size, max_len - y.shape[1])) * WordSequence.END), axis=1)

            targets = np.array(([
                (1, 0)
            ] * len(a)) + ([
                (0, 1)
            ] * len(a)))

            batch = np.concatenate((a, y), axis=0)
            batchl = np.concatenate((al, yl), axis=0)

            batch = batch.tolist()
            batchl = batchl.tolist()

            # batch, batchl = shuffle(batch, batchl)

            xx = np.concatenate((x, x), axis=0)
            xxl = np.concatenate((xl, xl), axis=0)

            # tmp_batch = list(zip(xx, xxl, batch, batchl))
            # tmp_batch = sorted(tmp_batch, key=lambda x: x[1], reverse=True)
            # xx, xxl, batch, batchl = zip(*tmp_batch)

            batch = np.array(batch).astype(np.int32)
            batchl = np.array(batchl)

            cost, acc = model_d.train(sess_d, xx, xxl, batch, batchl, targets)
            costs.append(cost)
            accuracy.append(acc)

            # print(batch, batchl)

            bar.set_description('epoch {} loss={:.6f} acc={:.6f} {}'.format(
                epoch,
                np.mean(costs),
                np.mean(accuracy),
                len(costs)
            ))

        model_d.save(sess_d, save_path)



def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
