"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import random
import itertools
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from rnn_crf import RNNCRF
from data_utils import batch_flow
from fake_data import generate


def test(bidirectional, cell_type, depth,
         use_residual, use_dropout, output_project_active, crf_loss):
    """测试不同参数在生成的假数据上的运行结果"""

    # 获取一些假数据
    x_data, y_data, ws_input, ws_target = generate(size=10000, same_len=True)

    # 训练部分

    split = int(len(x_data) * 0.8)
    x_train, x_test, y_train, y_test = (
        x_data[:split], x_data[split:], y_data[:split], y_data[split:])
    n_epoch = 1
    batch_size = 32
    steps = int(len(x_train) / batch_size) + 1

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = '/tmp/s2ss_crf.ckpt'

    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:

            model = RNNCRF(
                input_vocab_size=len(ws_input),
                target_vocab_size=len(ws_target),
                max_decode_step=100,
                batch_size=batch_size,
                learning_rate=0.001,
                bidirectional=bidirectional,
                cell_type=cell_type,
                depth=depth,
                use_residual=use_residual,
                use_dropout=use_dropout,
                output_project_active=output_project_active,
                hidden_units=64,
                embedding_size=64,
                parallel_iterations=1,
                crf_loss=crf_loss
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            # print(sess.run(model.input_layer.kernel))
            # exit(1)

            for epoch in range(1, n_epoch + 1):
                costs = []
                flow = batch_flow(
                    [x_train, y_train], [ws_input, ws_target], batch_size
                )
                bar = tqdm(range(steps),
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    cost = model.train(sess, x, xl, y, yl)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f}'.format(
                        epoch,
                        np.mean(costs)
                    ))

            model.save(sess, save_path)

    # 测试部分
    tf.reset_default_graph()
    model_pred = RNNCRF(
        input_vocab_size=len(ws_input),
        target_vocab_size=len(ws_target),
        max_decode_step=100,
        batch_size=batch_size,
        mode='decode',
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        use_residual=use_residual,
        use_dropout=use_dropout,
        output_project_active=output_project_active,
        hidden_units=64,
        embedding_size=64,
        parallel_iterations=1,
        crf_loss=crf_loss
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        flow = batch_flow([x_test, y_test], [ws_input, ws_target], batch_size)
        pbar = tqdm(range(100))
        acc = []
        for i in pbar:
            x, xl, y, yl = next(flow)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )

            for j in range(batch_size):
                right = np.sum(y[j][:yl[j]] == pred[j][:yl[j]])
                acc.append(right / yl[j])

            if i < 3:
                print(ws_input.inverse_transform(x[0]))
                print(ws_target.inverse_transform(y[0]))
                print(ws_target.inverse_transform(pred[0]))
            else:
                pbar.set_description('acc: {}'.format(np.mean(acc)))


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)

    params = OrderedDict((
        ('bidirectional', (True, False)),
        ('cell_type', ('gru', 'lstm')),
        ('depth', (1, 2, 3)),
        ('use_residual', (True, False)),
        ('use_dropout', (True, False)),
        ('output_project_active', (None, 'tanh', 'sigmoid', 'linear')),
        ('crf_loss', (False, True))
    ))

    loop = itertools.product(*params.values())

    for param_value in loop:
        param = OrderedDict(zip(params.keys(), param_value))
        print('=' * 30)
        for key, value in param.items():
            print(key, ':', value)
        print('-' * 30)
        test(**param)


if __name__ == '__main__':
    main()
