"""
对RNNCRF模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         use_residual, use_dropout, time_major, hidden_units,
         output_project_active, crf_loss=True, save_path='./s2ss_crf.ckpt'):
    """测试不同参数在生成的假数据上的运行结果"""

    from rnn_crf import RNNCRF
    from data_utils import batch_flow
    from word_sequence import WordSequence # pylint: disable=unused-variable

    x_data, y_data, ws_input, ws_target = pickle.load(
        open('ner.pkl', 'rb'))

    # 训练部分
    split = int(len(x_data) * 0.8)
    x_train, x_test, y_train, y_test = (
        x_data[:split], x_data[split:], y_data[:split], y_data[split:])
    n_epoch = 10
    batch_size = 128
    steps = int(len(x_train) / batch_size) + 1

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

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
                parallel_iterations=64,
                hidden_units=hidden_units,
                optimizer='adam',
                time_major=time_major,
                output_project_active=output_project_active,
                crf_loss=crf_loss
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            # print(sess.run(model.input_layer.kernel))
            # exit(1)

            flow = batch_flow(
                [x_train, y_train], [ws_input, ws_target], batch_size
            )

            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps), total=steps,
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
        hidden_units=hidden_units,
        time_major=time_major,
        parallel_iterations=1,
        output_project_active=output_project_active,
        crf_loss=crf_loss
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        pbar = tqdm(range(100))
        flow = batch_flow([x_test, y_test], [ws_input, ws_target], batch_size)
        acc = []
        prec = []
        rec = []
        for i in pbar:
            x, xl, y, yl = next(flow)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )

            for j in range(batch_size):

                right = np.asarray(ws_target.inverse_transform(y[j]))
                predict = ws_target.inverse_transform(pred[j])
                if len(predict) < len(right):
                    predict += ['O'] * (len(right) - len(predict))
                predict = np.asarray(predict)

                pp = predict[:yl[j]]
                rr = right[:yl[j]]
                if len(rr) > 0:
                    acc.append(np.sum(pp == rr) / len(rr))

                pp = predict[:yl[j]]
                rr = right[:yl[j]]
                pp = pp[rr != 'O']
                rr = rr[rr != 'O']
                if len(rr) > 0:
                    rec.append(np.sum(pp == rr) / len(rr))

                pp = predict[:yl[j]]
                rr = right[:yl[j]]
                rr = rr[pp != 'O']
                pp = pp[pp != 'O']
                if len(rr) > 0:
                    prec.append(np.sum(pp == rr) / len(rr))

            if i < 3:
                # print(ws_input.inverse_transform(x[0]))
                # print(ws_target.inverse_transform(y[0]))
                # print(ws_target.inverse_transform(pred[0]))
                pass
            else:
                pbar.set_description(
                    'acc: {:.4f} prec: {:.4f} rec: {:.4f}'.format(
                        np.mean(acc), np.mean(prec), np.mean(rec)))


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 1, False, True, False, 64, 'tanh')


if __name__ == '__main__':
    main()
