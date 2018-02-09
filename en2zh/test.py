"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import nltk
import numpy as np
import tensorflow as tf

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from sequence_to_sequence import batch_flow
    from word_sequence import WordSequence # pylint: disable=unused-variable

    x_data, _, ws_input, ws_target = pickle.load(open('en-zh_cn.pkl', 'rb'))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = '/tmp/s2ss_en2zh/'

    # 测试部分
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws_input),
        target_vocab_size=len(ws_target),
        batch_size=1,
        mode='decode',
        beam_width=12,
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        attention_type=attention_type,
        use_residual=use_residual,
        use_dropout=use_dropout,
        parallel_iterations=1,
        time_major=time_major,
        hidden_units=hidden_units # for test
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            user_text = input('Input English Sentence:')
            if user_text in ('exit', 'quit'):
                exit(0)
            x_test = [nltk.word_tokenize(user_text.lower())]
            bar = batch_flow(x_test, x_test, ws_input, ws_target, 1)
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


    # x_data, y_data, ws_input, ws_target = pickle.load(open('data.pkl', 'rb'))
    #
    # # 获取一些假数据
    # # x_data, y_data, ws_input, ws_target = generate(size=10000)
    #
    # # 训练部分
    #
    # split = int(len(x_data) * 0.8)
    # _, x_test, _, y_test = (
    #     x_data[:split], x_data[split:], y_data[:split], y_data[split:])
    #
    # config = tf.ConfigProto(
    #     device_count={'CPU': 1, 'GPU': 0},
    #     allow_soft_placement=True,
    #     log_device_placement=False
    # )
    #
    # save_path = '/tmp/s2ss_en2zh/'
    #
    # # 测试部分
    # tf.reset_default_graph()
    # model_pred = SequenceToSequence(
    #     input_vocab_size=len(ws_input),
    #     target_vocab_size=len(ws_target),
    #     batch_size=1,
    #     mode='decode',
    #     beam_width=5,
    #     bidirectional=bidirectional,
    #     cell_type=cell_type,
    #     depth=depth,
    #     attention_type=attention_type,
    #     use_residual=use_residual,
    #     use_dropout=use_dropout,
    #     parallel_iterations=1,
    #     hidden_units=128 # for test
    # )
    # init = tf.global_variables_initializer()
    #
    # with tf.Session(config=config) as sess:
    #     sess.run(init)
    #     model_pred.load(sess, save_path)
    #
    #     bar = batch_flow(x_test, y_test, ws_input, ws_target, 1)
    #     t = 0
    #     for x, xl, y, _ in bar:
    #         pred = model_pred.predict(
    #             sess,
    #             np.array(x),
    #             np.array(xl)
    #         )
    #         print(ws_input.inverse_transform(x[0]))
    #         print(ws_target.inverse_transform(y[0]))
    #         print(ws_target.inverse_transform(pred[0, :, 0]))
    #         t += 1
    #         if t >= 3:
    #             break
    #
    # tf.reset_default_graph()
    # model_pred = SequenceToSequence(
    #     input_vocab_size=len(ws_input),
    #     target_vocab_size=len(ws_target),
    #     batch_size=1,
    #     mode='decode',
    #     beam_width=1,
    #     bidirectional=bidirectional,
    #     cell_type=cell_type,
    #     depth=depth,
    #     attention_type=attention_type,
    #     use_residual=use_residual,
    #     use_dropout=use_dropout,
    #     parallel_iterations=1,
    #     hidden_units=128 # for test
    # )
    # init = tf.global_variables_initializer()
    #
    # with tf.Session(config=config) as sess:
    #     sess.run(init)
    #     model_pred.load(sess, save_path)
    #
    #     bar = batch_flow(x_test, y_test, ws_input, ws_target, 1)
    #     t = 0
    #     for x, xl, y, _ in bar:
    #         pred = model_pred.predict(
    #             sess,
    #             np.array(x),
    #             np.array(xl)
    #         )
    #         print(ws_input.inverse_transform(x[0]))
    #         print(ws_target.inverse_transform(y[0]))
    #         print(ws_target.inverse_transform(pred[0]))
    #         t += 1
    #         if t >= 3:
    #             break


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', False, True, True, 64)


if __name__ == '__main__':
    main()
