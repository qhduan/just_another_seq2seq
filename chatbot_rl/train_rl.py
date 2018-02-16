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
    from data_utils import batch_flow
    from data_utils import batch_flow_bucket_rl
    from word_sequence import WordSequence # pylint: disable=unused-variable

    dull_data = [
        '我不知道你在说什么',
        '你知道我的意思',
        '我们要去哪里？',
        '你什么都不知道',
        '你知道你知道',
        '我知道你是谁',
        '你说什么？',
        '为什么？',
        '我不知道',
        '我很抱歉',
        '我知道了',
        '不喜欢',
        '是好的',
        '是的',
        '不是',
        '我不',
        '好的',
        '什么',
        '没有',
        '喜欢',
        '我想',
        '什么',
        '不好',
        '我的',
        '我吧',
        '谁？',
        '是吗',
        '等等',
        '谢谢',
        '好的',
        '好',
        '是',
        '对',
        '我',
        '你',
        '不',
    ]

    dull_data = [jieba.lcut(x) for x in dull_data]

    p1_data, q1_data, p2_data, ws = pickle.load(
        open('chatbot_rl.pkl', 'rb'))

    # 训练部分
    n_epoch = 10
    batch_size = 512
    limit = 8
    lambda_1 = 0.25
    lambda_2 = 0.25
    lambda_3 = 0.5
    steps = int(len(p1_data) / batch_size) + 1

    dull_flow = []
    for dull in dull_data:
        d, dl, _, _ = next(batch_flow([dull], [dull], ws, ws, 1))
        dull_flow.append((
            np.array(d.tolist() * batch_size),
            np.array(dl.tolist() * batch_size)
        ))

    config = tf.ConfigProto(
        # device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './s2ss_chatbot_rl.ckpt'
    backward_path = './s2ss_chatbot_backward.ckpt'
    forward_path = './s2ss_chatbot_forward.ckpt'

    graph_backward = tf.Graph()
    graph_rl = tf.Graph()

    # 读取反向模型 seq2seq(x|y)
    with graph_backward.as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        sess_backward = tf.Session(config=config)

        model_backward = SequenceToSequence(
            input_vocab_size=len(ws),
            target_vocab_size=len(ws),
            batch_size=batch_size,
            learning_rate=0.01,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=32,
            hidden_units=hidden_units,
            optimizer='sgd',
            time_major=time_major
        )
        init = tf.global_variables_initializer()
        sess_backward.run(init)
        model_backward.load(sess_backward, backward_path)

    # 构建要强化徐诶训练的模型
    with graph_rl.as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        sess_rl = tf.Session(config=config)

        model_rl = SequenceToSequence(
            input_vocab_size=len(ws),
            target_vocab_size=len(ws),
            batch_size=batch_size,
            learning_rate=0.01,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=32,
            hidden_units=hidden_units,
            optimizer='sgd',
            time_major=time_major
        )
        init = tf.global_variables_initializer()
        sess_rl.run(init)
        if preload:
            model_rl.load(sess_rl, forward_path)

    # 开始训练
    flow = batch_flow_bucket_rl(
        p1_data, q1_data, p2_data, ws, batch_size
    )

    for epoch in range(1, n_epoch + 1):
        costs = []
        lengths = []
        bar = tqdm(range(steps), total=steps,
                   desc='epoch {}, loss=0.000000'.format(epoch))
        for _ in bar:

            p1, p1l, q1, q1l, p1q1, p1q1l, p2, p2l = next(flow)

            _, a = model_rl.entropy(
                sess_rl, p1q1, p1q1l, p2, p2l
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
            max_a_len = max([len(aa) for aa in new_a])
            a = []
            for aa in new_a:
                if len(aa) < max_a_len:
                    aa += [WordSequence.END] * (max_a_len - len(aa))
                a.append(aa)
            al = np.array(al)
            a = np.array(a)

            # print([ws.inverse_transform(aa) for aa in p1q1])
            # print([ws.inverse_transform(aa) for aa in a])

            # Ease of answering
            reward_1_s = None

            for d, dl in dull_flow:

                reward_1, _ = model_rl.entropy(
                    sess_rl, a, al, d, dl
                )

                reward_1 = np.sum(reward_1, axis=1)
                reward_1 *= -1.0 / len(d)
                if reward_1_s is None:
                    reward_1_s = reward_1
                else:
                    reward_1_s += reward_1

            reward_1_s = np.array(reward_1_s)
            reward_1 = reward_1_s / len(dull_flow)

            # print('reward_1.shape, reward_1', reward_1.shape, reward_1)
            # exit(0)

            # Information Flow
            # shape = (batch_size, time_step, embedding_size)
            emb_p1 = model_rl.get_encoder_embedding(sess_rl, p1)
            emb_p2 = model_rl.get_encoder_embedding(sess_rl, p2)

            shrink_size = min(emb_p1.shape[1], emb_p2.shape[1])

            emb_p1 = emb_p1[:, :shrink_size, :]
            emb_p2 = emb_p2[:, :shrink_size, :]

            def cos_distance(a, b):
                return np.sum(a * b) / (
                    np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

            reward_2 = np.array([
                (1 + cos_distance(emb_p1[i], emb_p2[i])) / 2
                for i in range(batch_size)
            ])
            reward_2 = -np.log(reward_2 + 1e-12)

            # print('reward_2.shape, reward_2', reward_2.shape, reward_2)

            # (a | qi, pi)
            forward_loss, _ = model_rl.entropy(
                sess_rl, p1q1, p1q1l, a, al
            )
            # (qi | a)
            backward_loss, _ = model_backward.entropy(
                sess_backward, a, al, q1, q1l
            )

            forward_loss = np.sum(forward_loss, axis=1) / al
            backward_loss = np.sum(backward_loss, axis=1) / q1l
            reward_3 = forward_loss + backward_loss

            # print('reward_3.shape, reward_3', reward_3.shape, reward_3)
            # exit(0)

            rewards = reward_1 * lambda_1 \
                + reward_2 * lambda_2 \
                + reward_3 * lambda_3
            # print('rewards.shape', rewards.shape)
            rewards = np.nan_to_num(rewards)
            rewards[rewards < 0] = 0
            rewards[rewards > 10] = 10
            rewards = rewards / 5
            rewards = rewards.reshape(-1, 1)

            cost = model_rl.train(sess_rl, p1q1, p1q1l, p2, p2l, rewards)

            costs.append(cost)
            lengths.append(np.mean(al))
            bar.set_description('epoch {} loss={:.6f} c={:.4f} r1={:.4f} r2={:.4f} r3={:.4f} rs={:.4f}'.format(
                epoch,
                np.mean(costs),
                np.mean(lengths),
                np.mean(reward_1),
                np.mean(reward_2),
                np.mean(reward_3),
                np.mean(rewards)
            ))

        # epoch end
        if limit > 0:
            limit -= 2
        if limit < 0:
            limit = 0
        model_rl.save(sess_rl, save_path)


def main():
    """入口程序，开始测试不同参数组合"""
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(True, 'lstm', 2, 'Bahdanau', True, True, True, 256)


if __name__ == '__main__':
    main()
