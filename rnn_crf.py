"""
QHDuan
2018-02-05

RNN-CRF Model

crf:
https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/crf
"""


import math

import numpy as np
import tensorflow as tf
# from tensorflow import layers
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import ResidualWrapper
from tensorflow.contrib.rnn import LSTMStateTuple

from word_sequence import WordSequence
from data_utils import _get_embed_device


class RNNCRF(object):
    """SequenceToSequence Model

    基本流程
    __init__ 基本参数保存，验证参数合法性
        build_model 开始构建整个模型
            init_placeholders 初始化一些tensorflow的变量占位符
            build_encoder 初始化编码器
                build_single_cell
                    build_encoder_cell
            build_decoder_crf 初始化解码器
            init_optimizer 如果是在训练模式则初始化优化器
    train 训练一个batch的数据
    predict 预测一个batch的数据
    """

    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 max_decode_step,
                 batch_size=32,
                 embedding_size=128,
                 mode='train',
                 hidden_units=256,
                 depth=1,
                 cell_type='lstm',
                 dropout=0.2,
                 use_dropout=False,
                 use_residual=False,
                 optimizer='adam',
                 learning_rate=0.001,
                 min_learning_rate=1e-6,
                 decay_steps=500000,
                 max_gradient_norm=5.0,
                 bidirectional=False,
                 output_project_active=None,
                 time_major=False,
                 seed=0,
                 parallel_iterations=32,
                 crf_loss=True):
        """保存参数变量，开始构建整个模型
        Args:
            input_vocab_size: 输入词表大小
            target_vocab_size: 输出词表大小
            max_decode_step:
                最大的解码长度，可以是很大的整数，默认是None
                None的情况下默认是encoder输入最大长度的 4 倍
            batch_size: 数据batch的大小
            embedding_size, 输入词表与输出词表embedding的维度
            mode: 取值为 train 或者 decode，训练模式或者预测模式
            hidden_units:
                RNN模型的中间层大小，encoder和decoder层相同
                如果encoder层是bidirectional的话，decoder层是双倍大小
            depth: encoder和decoder的rnn层数
            cell_type: rnn神经元类型，lstm 或者 gru
            dropout: dropout比例，取值 [0, 1)
            use_dropout: 是否使用dropout
            use_residual:# 是否使用residual
            optimizer: 优化方法， adam, adadelta, sgd, rmsprop, momentum
            learning_rate: 学习率
            max_gradient_norm: 梯度正则剪裁的系数
            bidirectional: encoder 是否为双向
            output_project_active:
                是否在crf之前使用一个投影层，并指定一个激活函数
                None, 'tanh', 'sigmoid', 'linear'
            time_major:
                是否在“计算过程”中使用时间为主的批量数据
                注意，改变这个参数并不要求改变输入数据的格式
                输入数据的格式为 [batch_size, time_step] 是一个二维矩阵
                time_step是句子长度
                经过 embedding 之后，数据会变为
                [batch_size, time_step, embedding_size]
                这是一个三维矩阵（或者三维张量Tensor）
                这样的数据格式是 time_major=False 的
                如果设置 time_major=True 的话，在部分计算的时候，会把矩阵转置为
                [time_step, batch_size, embedding_size]
                也就是 time_step 是第一维，所以叫 time_major
                TensorFlow官方文档认为time_major=True会比较快
            seed: 一些层间操作的随机数 seed 设置
            parallel_iterations:
                dynamic_rnn 和 dynamic_decode 的并行数量
                如果要取得可重复结果，在有dropout的情况下，应该设置为 1，否则结果会不确定
        """

        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_decode_step = max_decode_step
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.cell_type = cell_type
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.mode = mode
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_steps = decay_steps
        self.max_gradient_norm = max_gradient_norm
        self.keep_prob = 1.0 - dropout
        self.bidirectional = bidirectional
        self.output_project_active = output_project_active
        self.seed = seed
        self.parallel_iterations = parallel_iterations
        self.time_major = time_major
        self.crf_loss = crf_loss

        assert output_project_active in (None, 'tanh', 'sigmoid', 'linear'), \
            'output_project_active 必须是 None, "tanh", "sigmoid", "linear"之一'

        assert mode in ('train', 'decode'), \
            'mode 必须是 "train" 或 "decode" 而不是 "{}"'.format(mode)

        assert dropout >= 0.0 and dropout < 1.0, '0 <= dropout < 1'

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name='keep_prob'
        )

        self.global_step = tf.Variable(
            0, trainable=False, name='global_step'
        )
        self.global_epoch_step = tf.Variable(
            0, trainable=False, name='global_epoch_step'
        )
        self.global_epoch_step_op = tf.assign(
            self.global_epoch_step,
            self.global_epoch_step + 1
        )

        assert self.optimizer.lower() in \
            ('adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'), \
            'optimizer 必须是下列之一： adadelta, adam, rmsprop, momentum, sgd'

        self.build_model()


    def build_model(self):
        """构建整个模型
        分别构建
        编码器（encoder）
        解码器（decoder）
        优化器（只在训练时构建，optimizer）
        """
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder_crf()

        if self.mode == 'train':
            self.init_optimizer()

        self.saver = tf.train.Saver()


    def init_placeholders(self):
        """初始化训练、预测所需的变量
        """

        # 编码器输入，shape=(batch_size, time_step)
        # 有 batch_size 句话，每句话是最大长度为 time_step 的 index 表示
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size, None),
            name='encoder_inputs'
        )

        # crf 是固定长度的
        self.encoder_inputs_length = tf.fill(
            dims=[self.batch_size],
            value=self.max_decode_step,
            name='encoder_inputs_length'
        )

        # 训练模式

        # 解码器输入，shape=(batch_size, time_step)
        self.decoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size, None),
            name='decoder_inputs'
        )

        # 解码器长度输入，shape=(batch_size,)
        self.decoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,),
            name='decoder_inputs_length'
        )

        self.decoder_start_token = tf.ones(
            shape=(self.batch_size, 1),
            dtype=tf.int32
        ) * WordSequence.START

        self.decoder_end_token = tf.ones(
            shape=(self.batch_size, 1),
            dtype=tf.int32
        ) * WordSequence.END

        # 实际训练的解码器输入，实际上是 start_token + decoder_inputs
        self.decoder_inputs_train = tf.concat([
            self.decoder_start_token,
            self.decoder_inputs
        ], axis=1)

        # 这个变量用来计算一个mask，用来对loss函数的反向传播进行修正
        # 这里需要 + 1，因为会自动给训练结果增加 end_token
        self.decoder_inputs_length_train = self.decoder_inputs_length + 1

        # 实际训练的解码器目标，实际上是 decoder_inputs + end_token
        self.decoder_targets_train = tf.concat([
            self.decoder_inputs,
            self.decoder_end_token
        ], axis=1)


    def build_single_cell(self, n_hidden, use_residual):
        """构建一个单独的rnn cell
        Args:
            n_hidden: 隐藏层神经元数量
            use_residual: 是否使用residual wrapper
        """
        cell_type = LSTMCell
        if self.cell_type.lower() == 'gru':
            cell_type = GRUCell
        cell = cell_type(n_hidden)

        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob_placeholder,
                seed=self.seed
            )
        if use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def build_encoder_cell(self):
        """构建一个单独的编码器cell
        """
        return MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])


    def build_encoder(self):
        """构建编码器
        """
        # print("构建编码器")
        with tf.variable_scope('encoder'):
            # 构建 encoder_cell
            self.encoder_cell = self.build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(
                -sqrt3, sqrt3, dtype=tf.float32
            )

            # 编码器的embedding
            with tf.device(_get_embed_device(self.input_vocab_size)):
                self.encoder_embeddings = tf.get_variable(
                    name='embedding',
                    shape=(self.input_vocab_size, self.embedding_size),
                    initializer=initializer,
                    dtype=tf.float32
                )

            # embedded之后的输入 shape = (batch_size, time_step, embedding_size)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]

            inputs = self.encoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs, (1, 0, 2))

            if not self.bidirectional:
                (
                    self.encoder_outputs,
                    self.encoder_last_state
                ) = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )
            else:
                self.encoder_cell_bw = self.build_encoder_cell()
                (
                    (encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)
                ) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.encoder_cell,
                    cell_bw=self.encoder_cell_bw,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )

                self.encoder_outputs = tf.concat(
                    (encoder_fw_outputs, encoder_bw_outputs), 2)

                # 在 bidirectional 的情况下合并 state
                # QHD
                # borrow from
                # https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/model_new.py
                # 对上面链接中的代码有修改，因为原代码没有考虑多层cell的情况(MultiRNNCell)
                if isinstance(encoder_fw_state[0], LSTMStateTuple):
                    # LSTM 的 cell
                    self.encoder_last_state = tuple([
                        LSTMStateTuple(
                            c=tf.concat((
                                encoder_fw_state[i].c,
                                encoder_bw_state[i].c
                            ), 1),
                            h=tf.concat((
                                encoder_fw_state[i].h,
                                encoder_bw_state[i].h
                            ), 1)
                        )
                        for i in range(len(encoder_fw_state))
                    ])
                elif isinstance(encoder_fw_state[0], tf.Tensor):
                    # GRU 的中间状态只有一个，所以类型是 tf.Tensor
                    # 分别合并(concat)就可以了
                    self.encoder_last_state = tuple([
                        tf.concat(
                            (encoder_fw_state[i], encoder_bw_state[i]),
                            1, name='bidirectional_concat_{}'.format(i)
                        )
                        for i in range(len(encoder_fw_state))
                    ])


    def build_decoder_crf(self):
        """构建crf解码器
        """

        with tf.variable_scope('decoder_crf'):
            encoder_outputs = self.encoder_outputs

            hidden_units = self.hidden_units
            if self.bidirectional:
                hidden_units *= 2

            encoder_outputs = tf.concat(encoder_outputs,
                                        axis=2)
            encoder_outputs = tf.reshape(encoder_outputs,
                                         [-1, hidden_units], name='crf_output')
            self.encoder_outputs = encoder_outputs

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(
                -sqrt3, sqrt3, dtype=tf.float32
            )

            if self.output_project_active is not None:
                proj_w = tf.get_variable('proj_w',
                                         [hidden_units, hidden_units],
                                         initializer=initializer)
                proj_b = tf.get_variable('proj_b', [hidden_units],
                                         initializer=tf.zeros_initializer())
                encoder_outputs = tf.nn.xw_plus_b(
                    encoder_outputs, proj_w, proj_b, name='proj_output')

                if self.output_project_active == 'tanh':
                    encoder_outputs = tf.tanh(encoder_outputs)
                elif self.output_project_active == 'sigmoid':
                    encoder_outputs = tf.sigmoid(encoder_outputs)

            # 把encoder的结果进行一次线性变换所需要的变量
            crf_w = tf.get_variable('crf_w',
                                    [hidden_units, self.target_vocab_size],
                                    initializer=initializer)
            crf_b = tf.get_variable('crf_b', [self.target_vocab_size],
                                    initializer=tf.zeros_initializer())

            outputs = tf.nn.xw_plus_b(encoder_outputs,
                                      crf_w, crf_b, name='crf_output')

            # crf 计算必须固定住max_decode_step
            self.logits = tf.reshape(
                outputs,
                shape=[self.batch_size,
                       self.max_decode_step,
                       self.target_vocab_size])

            if self.crf_loss:
                (
                    log_likelihood,
                    self.transition_params
                ) = tf.contrib.crf.crf_log_likelihood(
                    self.logits,
                    self.decoder_inputs,
                    self.decoder_inputs_length)

                (
                    self.viterbi_sequence,
                    self.viterbi_score
                ) = tf.contrib.crf.crf_decode(
                    self.logits,
                    self.transition_params,
                    self.encoder_inputs_length)

                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                self.outputs = tf.argmax(self.logits, 2)

                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.decoder_inputs, logits=self.logits)

                masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length,
                    maxlen=self.max_decode_step,
                    dtype=tf.float32, name='masks'
                )

                self.loss *= masks
                self.loss = tf.reduce_sum(self.loss)


    def save(self, sess, save_path='model.ckpt'):
        """保存模型"""
        self.saver.save(sess, save_path=save_path)


    def load(self, sess, save_path='model.ckpt'):
        """读取模型"""
        print('try load model from', save_path)
        self.saver.restore(sess, save_path)


    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """检查输入变量，并返回input_feed

        我们首先会把数据编码，例如把“你好吗”，编码为[0, 1, 2]
        多个句子组成一个batch，共同训练，例如一个batch_size=2，那么训练矩阵就可能是
        encoder_inputs = [
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]
        它所代表的可能是：[['我', '是', '帅', '哥'], ['你', '好', '啊', '</s>']]
        注意第一句的真实长度是 4，第二句只有 3（最后的</s>是一个填充数据）

        那么：
        encoder_inputs_length = [4, 3]
        来代表输入整个batch的真实长度
        注意，为了符合算法要求，每个batch的句子必须是长度降序的，也就是说你输入一个
        encoder_inputs_length = [1, 10] 这样是错误的，必须在输入前排序到
        encoder_inputs_length = [10, 1] 这样才行

        decoder_inputs 和 decoder_inputs_length 所代表的含义差不多

        Args:
            encoder_inputs:
                一个整形二维矩阵 [batch_size, max_source_time_steps]
            encoder_inputs_length:
                一个整形向量 [batch_size]
                每个维度是encoder句子的真实长度
            decoder_inputs:
                一个整形矩阵 [batch_size, max_target_time_steps]
            decoder_inputs_length:
                一个整形向量 [batch_size]
                每个维度是decoder句子的真实长度
            decode: 用来指示正在训练模式(decode=False)还是预测模式(decode=True)
        Returns:
            tensorflow所操作需要的input_feed，包括
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length
        """

        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError(
                "Encoder inputs and their lengths must be equal in their "
                "batch_size, %d != %d" % (
                    input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError(
                    "Encoder inputs and Decoder inputs must be equal in their "
                    "batch_size, %d != %d" % (
                        input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError(
                    "Decoder targets and their lengths must be equal in their "
                    "batch_size, %d != %d" % (
                        target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed


    def init_optimizer(self):
        """初始化优化器
        支持的方法有 sgd, adadelta, adam, rmsprop, momentum
        """

        # 学习率下降算法
        learning_rate = tf.train.polynomial_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.min_learning_rate,
            power=0.5
        )
        self.current_learning_rate = learning_rate

        # 设置优化器,合法的优化器如下
        # 'adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)


    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):
        """训练模型"""

        # 如果是crf模式，自动 padding 到 self.max_decode_step
        # self.max_decode_step 相当于 max_time_step
        encoder_inputs_crf = []
        for item in encoder_inputs:
            encoder_inputs_crf.append(list(item) + \
                [WordSequence.PAD] * (self.max_decode_step - len(item)))
        encoder_inputs = np.array(encoder_inputs_crf)

        decoder_inputs_crf = []
        for item in decoder_inputs:
            decoder_inputs_crf.append(list(item) + \
                [WordSequence.PAD] * (self.max_decode_step - len(item)))
        decoder_inputs = np.array(decoder_inputs_crf)

        # 输入
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )

        # 设置 dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        # 输出
        output_feed = [self.updates, self.loss]
        _, cost = sess.run(output_feed, input_feed)

        return cost


    def predict(self, sess,
                encoder_inputs, encoder_inputs_length):
        """预测输出"""

        # 输入
        # 如果是 crf 模式，就把输入补全到最大长度 self.max_decode_step
        # 相当于 max_time_step
        encoder_inputs_crf = []
        for item in encoder_inputs:
            encoder_inputs_crf.append(list(item) + \
                [WordSequence.PAD] * (self.max_decode_step - len(item)))
        encoder_inputs = np.array(encoder_inputs_crf)

        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            np.zeros(encoder_inputs.shape),
            np.zeros(encoder_inputs_length.shape),
            True
        )

        input_feed[self.keep_prob_placeholder.name] = 1.0

        # crf mode
        if self.crf_loss:
            pred = sess.run(self.viterbi_sequence, input_feed)
            preds = []
            for i in range(pred.shape[0]):
                item = pred[i][:encoder_inputs_length[i]]
                preds.append(item)

            return np.array(preds)
        # else:
        pred = sess.run(self.outputs, input_feed)
        preds = []
        for i in range(pred.shape[0]):
            item = pred[i][:encoder_inputs_length[i]]
            preds.append(item)

        return np.array(preds)
