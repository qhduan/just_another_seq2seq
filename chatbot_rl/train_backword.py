"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow_bucket_rl
    from word_sequence import WordSequence # pylint: disable=unused-variable

    p1_data, q1_data, p2_data, ws = pickle.load(
        open('chatbot_rl.pkl', 'rb'))

    # 训练部分
    n_epoch = 10
    batch_size = 256
    steps = int(len(p1_data) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './s2ss_chatbot_backward.ckpt'

    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:

            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                learning_rate=0.001,
                bidirectional=bidirectional,
                cell_type=cell_type,
                depth=depth,
                attention_type=attention_type,
                use_residual=use_residual,
                use_dropout=use_dropout,
                parallel_iterations=64,
                hidden_units=hidden_units,
                optimizer='adam',
                time_major=time_major
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            flow = batch_flow_bucket_rl(
                p1_data, q1_data, p2_data, ws, batch_size
            )

            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    _, _, q1, q1l, _, _, p2, p2l = next(flow)
                    # cost = model.train(sess, x, xl, y, yl)
                    # train backward
                    cost = model.train(sess, p2, p2l, q1, q1l)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f}'.format(
                        epoch,
                        np.mean(costs)
                    ))

            model.save(sess, save_path)

    # 测试部分
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=12,
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        attention_type=attention_type,
        use_residual=use_residual,
        use_dropout=use_dropout,
        hidden_units=hidden_units,
        time_major=time_major,
        parallel_iterations=1 # for test
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow_bucket_rl(
            p1_data, q1_data, p2_data, ws, 1
        )
        t = 0
        for _, _, q1, q1l, _, _, p2, p2l in bar:
            pred = model_pred.predict(
                sess,
                np.array(p2),
                np.array(p2l)
            )
            print(ws.inverse_transform(q1[0]))
            print(ws.inverse_transform(p2[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            if t >= 3:
                break


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
