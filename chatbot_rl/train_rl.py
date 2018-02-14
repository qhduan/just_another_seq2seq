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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow
    from data_utils import batch_flow_bucket_rl
    from word_sequence import WordSequence # pylint: disable=unused-variable

    dull_data = [
        list('我不知道你在说什么'),
        list('你知道我的意思'),
        list('你知道你知道'),
        list('你什么都不知道'),
        list('你说什么？'),
        list('为什么？'),
        list('我不知道'),
        list('我很抱歉'),
        list('我知道了'),
        list('不喜欢'),
        list('是的'),
        list('不是'),
        list('我不'),
        list('好的'),
        list('什么'),
        list('没有'),
        list('喜欢'),
        list('我想'),
        list('什么'),
        list('不好'),
        list('谁？'),
        list('好'),
    ]

    p1_data, q1_data, p2_data, ws = pickle.load(
        open('chatbot_rl.pkl', 'rb'))

    # 训练部分
    n_epoch = 5
    batch_size = 256
    padding_size = 20
    limit = 8
    lambda_1 = 0.25
    lambda_2 = 0.25
    lambda_3 = 0.5
    steps = int(len(p1_data) / batch_size) + 1

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
            learning_rate=0.001,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=32,
            hidden_units=hidden_units,
            optimizer='adam',
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
            learning_rate=0.001,
            bidirectional=bidirectional,
            cell_type=cell_type,
            depth=depth,
            attention_type=attention_type,
            use_residual=use_residual,
            use_dropout=use_dropout,
            parallel_iterations=32,
            hidden_units=hidden_units,
            optimizer='adam',
            time_major=time_major
        )
        init = tf.global_variables_initializer()
        sess_rl.run(init)
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
            # print('a.shape', a.shape, a)

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
            # print('al', al)
            # print('a', a)

            # Ease of answering

            reward_1_s = []
            for dull in dull_data:
                dull_flow = batch_flow([dull], [dull], ws, ws, 1)
                d, dl, _, _ = next(dull_flow)
                d = np.array(d.tolist() * batch_size)
                dl = np.array(dl.tolist() * batch_size)

                reward_1, _ = model_rl.entropy(
                    sess_rl, a, al, d, dl
                )
                reward_1 = -reward_1
                reward_1 = np.pad(reward_1,
                                  ((0, 0), (0, padding_size - reward_1.shape[1])),
                                  'constant', constant_values=0)
                reward_1 /= dl[0] + 1e-12
                reward_1_s.append(reward_1)
            reward_1_s = np.array(reward_1_s)
            reward_1 = np.sum(np.mean(reward_1_s, axis=0), axis=1)
            reward_1 = np.nan_to_num(reward_1)
            reward_1[reward_1 == -np.inf] = 0
            reward_1[reward_1 == np.inf] = 0
            # print(reward_1.shape, reward_1)

            # Information Flow
            # shape = (batch_size, time_step, embedding_size)
            emb_p1 = model_rl.get_encoder_embedding(sess_rl, p1)
            emb_p2 = model_rl.get_encoder_embedding(sess_rl, p2)

            emb_p1 = np.pad(emb_p1,
                            ((0, 0), (0, padding_size - emb_p1.shape[1]), (0, 0)),
                            'constant', constant_values=0)
            emb_p2 = np.pad(emb_p2,
                            ((0, 0), (0, padding_size - emb_p2.shape[1]), (0, 0)),
                            'constant', constant_values=0)

            # print('emb_p1.shape, emb_p2.shape', emb_p1.shape, emb_p2.shape)
            reward_2 = np.sum(emb_p1 * emb_p2, axis=(1, 2))) / (
                np.sqrt(np.sum(emb_p1 ** 2)) * np.sqrt(np.sum(emb_p2 ** 2)) \
                + 1e-12
            )
            reward_2 = -np.log(reward_2 + 1e-12)
            # reward_2 /= batch_size
            reward_2 = np.nan_to_num(reward_2)
            reward_2[reward_2 == -np.inf] = 0
            reward_2[reward_2 == np.inf] = 0
            # print(reward_2.shape, reward_2)

            # print('reward_2.shape', reward_2.shape)
            # print(reward_2)
            # print(p1q1.shape, p1q1l.shape, a.shape, al.shape)

            # (a | qi, pi)
            forward_loss, _ = model_rl.entropy(
                sess_rl, p1q1, p1q1l, a, al
            )
            # (qi | a)
            backward_loss, _ = model_backward.entropy(
                sess_backward, a, al, q1, q1l
            )

            forward_loss = np.pad(forward_loss,
                                  ((0, 0), (0, padding_size - forward_loss.shape[1])),
                                  'constant', constant_values=0)

            backward_loss = np.pad(backward_loss,
                                  ((0, 0), (0, padding_size - backward_loss.shape[1])),
                                  'constant', constant_values=0)

            # print('forward_loss.shape, backward_loss.shape', forward_loss.shape, backward_loss.shape)

            for i in range(forward_loss.shape[0]):
                forward_loss[i,:] /= al[i] + 1e-12

            for i in range(backward_loss.shape[0]):
                backward_loss[i,:] /= q1l[i] + 1e-12

            reward_3 = forward_loss + backward_loss
            reward_3 = np.nan_to_num(reward_3)
            reward_3[reward_3 == -np.inf] = 0
            reward_3[reward_3 == np.inf] = 0
            reward_3 = np.sum(reward_3, axis=1)

            # print(reward_3.shape, reward_3)
            # exit(0)

            # reward_1 = sigmoid(reward_1)
            # reward_2 = sigmoid(reward_2)
            # reward_3 = sigmoid(reward_3)

            # print(np.mean(reward_1), np.mean(reward_2), np.mean(reward_3))

            rewards = reward_1 * lambda_1 + reward_2 * lambda_2 + reward_3 * lambda_3

            # gradually anneal the value of L to zero
            # rewards[:,:limit] = 1.0

            # print('rewards.shape', rewards.shape)

            # +1 是因为 END 符号
            # rewards = rewards[:, :p2.shape[1] + 1]
            rewards = np.nan_to_num(rewards)
            rewards[rewards == -np.inf] = 0
            rewards[rewards == np.inf] = 0
            rewards = rewards.reshape(-1, 1)

            # rewards[rewards > 5] = 5
            # rewards[rewards < 0] = 0

            # print(rewards.shape, rewards)
            # exit(0)

            # print('rewards.shape', rewards.shape, rewards)
            #
            # print('q1.shape, q1l.shape, p2.shape, p2l.shape', q1.shape, q1l.shape, p2.shape, p2l.shape)

            cost = model_rl.train(sess_rl, p1q1, p1q1l, p2, p2l, rewards)

            costs.append(cost)
            lengths.append(np.mean(al))
            bar.set_description('epoch {} loss={:.6f} {:.4f}'.format(
                epoch,
                np.mean(costs),
                np.mean(lengths)
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
    test(True, 'lstm', 1, 'Bahdanau', True, True, True, 64)


if __name__ == '__main__':
    main()
