
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

    # 训练部分
    n_epoch = 10
    batch_size = 128
    steps = int(len(x1_data) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './s2ss_chatbot_samperson.ckpt'

    graph = tf.Graph()

    with graph.as_default():
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
            parallel_iterations=32,
            time_major=time_major,
            hidden_units=hidden_units,
            optimizer='adam'
        )
        init = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init)
        # model.load(sess, save_path)

    # 开始训练
    flow = batch_flow([x1_data, x2_data, y_data], [ws, ws, None], batch_size)

    for epoch in range(1, n_epoch + 1):
        costs = []
        accuracy = []
        bar = tqdm(range(steps), total=steps,
                   desc='epoch {}, loss=0.000000'.format(epoch))
        for _ in bar:

            x1, x1l, x2, x2l, y, _ = next(flow)

            cost, acc = model.train(sess, x1, x1l, x2, x2l, y)
            costs.append(cost)
            accuracy.append(acc)

            # ret = model.predict(sess, x1, x1l, x2, x2l)
            # print(ret)

            # print(batch, batchl)

            bar.set_description('epoch {} loss={:.6f} acc={:.6f} {}'.format(
                epoch,
                np.mean(costs),
                np.mean(accuracy),
                len(costs)
            ))

        model.save(sess, save_path)



def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
