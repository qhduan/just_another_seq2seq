"""
对RNNCRF模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         use_residual, use_dropout, time_major,
         hidden_units, output_project_active):
    """测试不同参数在生成的假数据上的运行结果"""

    from rnn_crf import RNNCRF
    from data_utils import batch_flow
    from word_sequence import WordSequence # pylint: disable=unused-variable

    x_data, _, ws_input, ws_target = pickle.load(open('ner.pkl', 'rb'))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './s2ss_crf.ckpt'

    # 测试部分
    tf.reset_default_graph()
    model_pred = RNNCRF(
        input_vocab_size=len(ws_input),
        target_vocab_size=len(ws_target),
        max_decode_step=100,
        batch_size=1,
        mode='decode',
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        use_residual=use_residual,
        use_dropout=use_dropout,
        parallel_iterations=1,
        time_major=time_major,
        hidden_units=hidden_units,
        output_project_active=output_project_active
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            user_text = input('Input Sentence:')
            if user_text in ('exit', 'quit'):
                exit(0)
            x_test = [list(user_text.lower())]
            bar = batch_flow([x_test, x_test], [ws_input, ws_target], 1)
            x, xl, _, _ = next(bar)
            # x = np.array([
            #     list(reversed(xx))
            #     for xx in x
            # ])
            print(x, xl)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(pred)
            print(ws_input.inverse_transform(x[0]))
            print(ws_target.inverse_transform(pred[0]))


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 1, False, True, False, 64, 'tanh')


if __name__ == '__main__':
    main()
