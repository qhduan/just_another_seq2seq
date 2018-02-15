"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow
    from word_sequence import WordSequence # pylint: disable=unused-variable

    _, _, _, ws = pickle.load(open('chatbot_rl.pkl', 'rb'))

    # for x in x_data[:5]:
    #     print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    # save_path = '/tmp/s2ss_chatbot.ckpt'
    save_path = './s2ss_chatbot_forward.ckpt'
    save_path_rl = './s2ss_chatbot_rl.ckpt'

    graph = tf.Graph()
    graph_rl = tf.Graph()

    # 测试部分
    with graph.as_default():
        model_pred = SequenceToSequence(
            input_vocab_size=len(ws),
            target_vocab_size=len(ws),
            batch_size=1,
            mode='decode',
            beam_width=0,
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
        sess = tf.Session(config=config)
        sess.run(init)
        model_pred.load(sess, save_path)

    with graph_rl.as_default():
        model_rl = SequenceToSequence(
            input_vocab_size=len(ws),
            target_vocab_size=len(ws),
            batch_size=1,
            mode='decode',
            beam_width=0,
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
        sess_rl = tf.Session(config=config)
        sess_rl.run(init)
        model_rl.load(sess, save_path_rl)


    last = None

    while True:
        user_text = input('Input Chat Sentence:')
        if user_text in ('exit', 'quit'):
            exit(0)
        x_test = list(user_text.lower())
        # if last is not None and last:
        #     print(last)
        #     x_test = last + [WordSequence.PAD_TAG] + x_test
        x_test = [x_test]
        bar = batch_flow(x_test, x_test, ws, ws, 1)
        x, xl, _, _ = next(bar)
        print(x, xl)
        pred = model_pred.predict(
            sess,
            np.array(x),
            np.array(xl)
        )
        pred_rl = model_pred.predict(
            sess,
            np.array(x),
            np.array(xl)
        )
        print(ws.inverse_transform(x[0]))
        print('no:', ws.inverse_transform(pred[0]))
        print('rl:', ws.inverse_transform(pred_rl[0]))
        p = []
        for pp in ws.inverse_transform(pred_rl[0]):
            if pp == WordSequence.END_TAG:
                break
            if pp == WordSequence.PAD_TAG:
                break
            p.append(pp)
        last = p


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
