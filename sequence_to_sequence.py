"""
QHDuan
2018-02-05

sequence to sequence Model

官方文档的意思好像是time_major=True的情况下会快一点
https://www.tensorflow.org/tutorials/seq2seq
不过现在代码都在time_major=False上

test on tensorflow == 1.4.1
seq2seq:
https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/seq2seq

Code was borrow heavily from:
https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py
Another wonderful example is:
https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm
Official sequence2sequence project:
https://github.com/tensorflow/nmt
Another official sequence2sequence model:
https://github.com/tensorflow/tensor2tensor
"""


import math
import random

import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import ResidualWrapper
from tensorflow.contrib.rnn import LSTMStateTuple

from word_sequence import WordSequence
from data_utils import _get_embed_device


class SequenceToSequence(object):
    """SequenceToSequence Model

    基本流程
    __init__ 基本参数保存，验证参数合法性
        build_model 开始构建整个模型
            init_placeholders 初始化一些tensorflow的变量占位符
            build_encoder 初始化编码器
                build_single_cell
                    build_encoder_cell
            build_decoder 初始化解码器
                build_single_cell
                    build_decoder_cell
            init_optimizer 如果是在训练模式则初始化优化器
    train 训练一个batch的数据
    predict 预测一个batch的数据
    """

    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 batch_size=32,
                 embedding_size=128,
                 mode='train',
                 hidden_units=256,
                 depth=1,
                 attn_input_feeding=False,
                 beam_width=0,
                 cell_type='lstm',
                 dropout=0.2,
                 use_dropout=False,
                 use_residual=False,
                 optimizer='adam',
                 learning_rate=0.001,
                 max_gradient_norm=5.0,
                 max_decode_step=None,
                 attention_type='Bahdanau',
                 bidirectional=False,
                 alignment_history=False,
                 time_major=False,
                 seed=0,
                 parallel_iterations=32):
        """保存参数变量，开始构建整个模型
        Args:
            input_vocab_size: 输入词表大小
            target_vocab_size: 输出词表大小
            batch_size: 数据batch的大小
            embedding_size, 输入词表与输出词表embedding的维度
            mode: 取值为 train 或者 decode，训练模式或者预测模式
            hidden_units:
                RNN模型的中间层大小，encoder和decoder层相同
                如果encoder层是bidirectional的话，decoder层是双倍大小
            depth: encoder和decoder的rnn层数
            attn_input_feeding: 输入给 attention 层的时候，是否使用一个投影层
            beam_width:
                beam_width是beamsearch的超参，用于解码
                如果大于0则使用beamsearch，小于等于0则不使用
            cell_type: rnn神经元类型，lstm 或者 gru
            dropout: dropout比例，取值 [0, 1)
            use_dropout: 是否使用dropout
            use_residual:# 是否使用residual
            optimizer: 优化方法， adam, adadelta, sgd, rmsprop, momentum
            learning_rate: 学习率
            max_gradient_norm: 梯度正则剪裁的系数
            max_decode_step:
                最大的解码长度，可以是很大的整数，默认是None
                None的情况下默认是encoder输入最大长度的 4 倍
            attention_type: 'Bahdanau' or 'Luong' 不同的 attention 类型
            bidirectional: encoder 是否为双向
            alignment_history:
                是否记录alignment历史，用于查看attention热力图
                详情可以参考 test_atten.py 文件中显示热力图的代码
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
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.attn_input_feeding = attn_input_feeding
        self.cell_type = cell_type
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.mode = mode
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.keep_prob = 1.0 - dropout
        self.bidirectional = bidirectional
        self.seed = seed
        self.parallel_iterations = parallel_iterations
        self.time_major = time_major
        # Initialize encoder_embeddings to have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        self.initializer = tf.random_uniform_initializer(
            -sqrt3, sqrt3, dtype=tf.float32
        )

        assert mode in ('train', 'decode'), \
            'mode 必须是 "train" 或 "decode" 而不是 "{}"'.format(mode)

        assert dropout >= 0.0 and dropout < 1.0, '0 <= dropout < 1'

        assert attention_type.lower() in ('bahdanau', 'luong'), \
            '''attention_type 必须是 "bahdanau" 或 "luong" 而不是 "{}"
            '''.format(attention_type)

        assert beam_width < target_vocab_size, \
            'beam_width {} must < target vocab size {}'.format(
                beam_width, target_vocab_size
            )

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

        self.use_beamsearch_decode = False
        self.beam_width = beam_width
        self.use_beamsearch_decode = True if self.beam_width > 0 else False
        self.max_decode_step = max_decode_step

        self.alignment_history = alignment_history

        assert (self.use_beamsearch_decode and not self.alignment_history) or \
            (not self.use_beamsearch_decode and self.alignment_history) or \
            (not self.use_beamsearch_decode and not self.alignment_history), \
            'beamsearch和alignment_history不能同时打开'

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
        self.build_decoder()

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

        # 编码器长度输入，shape=(batch_size, 1)
        # 指的是 batch_size 句话每句话的长度
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,),
            name='encoder_inputs_length'
        )

        if self.mode == 'train':
            # 训练模式

            # 解码器输入，shape=(batch_size, time_step)
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size, None),
                name='decoder_inputs'
            )

            # 解码器输入的reward，用于强化学习训练，shape=(batch_size, time_step)
            self.rewards = tf.placeholder(
                dtype=tf.float32,
                shape=(self.batch_size, 1),
                name='rewards'
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

            # 编码器的embedding
            with tf.device(_get_embed_device(self.input_vocab_size)):
                self.encoder_embeddings = tf.get_variable(
                    name='embedding',
                    shape=(self.input_vocab_size, self.embedding_size),
                    initializer=self.initializer,
                    dtype=tf.float32
                )

            # embedded之后的输入 shape = (batch_size, time_step, embedding_size)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            # 输入投影层
            # 如果使用了residual，为了对齐输入和输出层，这里可能必须增加一个投影
            input_layer = layers.Dense(
                self.hidden_units, dtype=tf.float32, name='input_projection'
            )
            self.input_layer = input_layer

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(
                self.encoder_inputs_embedded
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



    def build_decoder_cell(self):
        """构建解码器cell"""

        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, (1, 0, 2))

        # To use BeamSearchDecoder
        # encoder_outputs, encoder_last_state, encoder_inputs_length
        # needs to be tiled so that:
        # [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:

            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width),
                self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        # 计算解码器的隐藏神经元数，如果编码器是 bidirectional 的
        # 那么解码器的一些隐藏神经元应该乘2
        num_units = self.hidden_units
        if self.bidirectional:
            num_units *= 2

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = BahdanauAttention(
            num_units=num_units,
            memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length
        )
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = LuongAttention(
                num_units=num_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )

        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell(
                num_units,
                use_residual=True
            )
            for i in range(self.depth)
        ]

        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):
            """根据attn_input_feeding属性来判断是否在attention计算前进行一次投影计算
            """
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            hidden_units = self.hidden_units
            if self.bidirectional:
                hidden_units *= 2
            input_layer = layers.Dense(hidden_units, dtype=tf.float32,
                                       name='attn_input_feeding')
            return input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            # attention_layer_size=self.hidden_units,
            attention_layer_size=num_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=self.alignment_history,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted
        # into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used,
        # the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        # 如果使用了 beamsearch 那么输入应该是 beam_width 倍于 batch_size 的
        batch_size = self.batch_size if not self.use_beamsearch_decode \
                     else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=tf.float32)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state


    def build_decoder(self):
        """构建解码器
        """
        with tf.variable_scope('decoder') as decoder_scope:
            # Building decoder_cell and decoder_initial_state
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell()

            # 解码器embedding
            with tf.device(_get_embed_device(self.target_vocab_size)):
                self.decoder_embeddings = tf.get_variable(
                    name='embeddings',
                    shape=(self.target_vocab_size, self.embedding_size),
                    initializer=self.initializer,
                    dtype=tf.float32
                )

            # On Using Very Large Target Vocabulary
            # for Neural Machine Translation
            # https://arxiv.org/pdf/1412.2007v2.pdf

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            hidden_units = self.hidden_units
            if self.bidirectional:
                hidden_units *= 2
            input_layer = layers.Dense(
                hidden_units,
                dtype=tf.float32,
                name='input_projection'
            )

            self.output_layer = layers.Dense(
                self.target_vocab_size, use_bias=False,
                name='output_projection')

            if self.mode == 'train':
                # decoder_inputs_embedded:
                # [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings,
                    ids=self.decoder_inputs_train
                )

                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded = input_layer(
                    self.decoder_inputs_embedded
                )

                # Helper to feed inputs for training:
                # read inputs from dense ground truth vectors
                inputs = self.decoder_inputs_embedded
                if self.time_major:
                    inputs = tf.transpose(inputs, (1, 0, 2))

                training_helper = seq2seq.TrainingHelper(
                    inputs=inputs,
                    sequence_length=self.decoder_inputs_length_train,
                    time_major=self.time_major,
                    name='training_helper'
                )

                # 训练的时候不在这里应用 output_layer
                # 因为这里会每个 time_step 的进行 output_layer 的投影计算，比较慢
                # 注意这个trick要成功必须设置 dynamic_decode 的 scope 参数
                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state,
                    # output_layer=self.output_layer
                )

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(
                    self.decoder_inputs_length_train
                )

                # decoder_outputs_train: BasicDecoderOutput
                #     namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output:
                #     if output_time_major=False:
                #         [batch_size, max_time_step + 1, num_decoder_symbols]
                #     if output_time_major=True:
                #         [max_time_step + 1, batch_size, num_decoder_symbols]
                # decoder_outputs_train.sample_id: [batch_size], tf.int32

                (
                    outputs,
                    self.final_state, # contain attention
                    _ # self.final_sequence_lengths
                ) = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=self.time_major,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                )

                # More efficient to do the projection
                # on the batch-time-concatenated tensor
                # logits_train:
                # [batch_size, max_time_step + 1, num_decoder_symbols]
                # 训练的时候一次性对所有的结果进行 output_layer 的投影运算
                # 官方NMT库说这样能提高10~20%的速度
                # 实际上我提高的速度会更大
                self.decoder_logits_train = self.output_layer(
                    outputs.rnn_output
                )

                # masks: masking for valid and padded time steps,
                # [batch_size, max_time_step + 1]
                self.masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length_train,
                    maxlen=max_decoder_length,
                    dtype=tf.float32, name='masks'
                )


                # Computes per word average cross-entropy over a batch
                # Internally calls
                # 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default

                decoder_logits_train = self.decoder_logits_train
                if self.time_major:
                    decoder_logits_train = tf.transpose(decoder_logits_train,
                                                        (1, 0, 2))

                self.decoder_pred_train = tf.argmax(
                    decoder_logits_train, axis=-1,
                    name='decoder_pred_train')

                # 下面的一些变量用于强化学习训练
                self.train_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.decoder_targets_train,
                    logits=decoder_logits_train)
                # self.train_entropy *= self.masks
                # print(self.train_entropy.shape)
                self.train_entropy_rewards = tf.multiply(self.train_entropy, self.rewards)
                # print('self.train_entropy_rewards.shape', self.train_entropy_rewards.shape)
                self.train_entropy_rewards *= self.masks

                # https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/seq2seq/python/ops/loss.py
                # if average_across_timesteps and average_across_batch:
                #   crossent = math_ops.reduce_sum(crossent)
                #   total_size = math_ops.reduce_sum(weights)
                #   total_size += 1e-12  # to avoid division by 0 for all-0 weights
                #   crossent /= total_size

                self.loss_without_rewards = tf.reduce_sum(self.train_entropy)
                self.loss_rewards = tf.reduce_sum(self.train_entropy_rewards)

                total_size = tf.reduce_sum(self.masks)
                total_size += 1e-12
                self.loss_without_rewards /= total_size
                self.loss_rewards /= total_size


                self.loss = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_targets_train,
                    weights=self.masks,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

            elif self.mode == 'decode':
                # 预测模式，非训练

                start_tokens = tf.fill(
                    [self.batch_size],
                    WordSequence.START
                )
                end_token = WordSequence.END

                def embed_and_input_proj(inputs):
                    """输入层的投影层wrapper
                    """
                    return input_layer(tf.nn.embedding_lookup(
                        self.decoder_embeddings,
                        inputs
                    ))

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding:
                    # uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=start_tokens,
                        end_token=end_token,
                        embedding=embed_and_input_proj
                    )
                    # Basic decoder performs greedy decoding at each time step
                    # print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(
                        cell=self.decoder_cell,
                        helper=decoding_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=self.output_layer
                    )
                else:
                    # Beamsearch is used to approximately
                    # find the most likely translation
                    # print("building beamsearch decoder..")
                    inference_decoder = BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.output_layer,
                    )

                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #     namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output:
                # if output_time_major=False:
                #     [batch_size, max_time_step, num_decoder_symbols]
                # if output_time_major=True
                #     [max_time_step, batch_size, num_decoder_symbols]
                # decoder_outputs_decode.sample_id:
                # if output_time_major=False
                #     [batch_size, max_time_step], tf.int32
                # if output_time_major=True
                #     [max_time_step, batch_size], tf.int32

                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #     namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids:
                # if output_time_major=False:
                #     [batch_size, max_time_step, beam_width]
                # if output_time_major=True
                #     [max_time_step, batch_size, beam_width]
                # decoder_outputs_decode.beam_search_decoder_output:
                #     BeamSearchDecoderOutput instance
                #     namedtuple(scores, predicted_ids, parent_ids)

                # 官方文档提到的一个潜在的最大长度选择
                # maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
                # https://www.tensorflow.org/tutorials/seq2seq

                if self.max_decode_step is not None:
                    max_decode_step = self.max_decode_step
                else:
                    # 默认 4 倍输入长度的输出解码
                    max_decode_step = tf.round(tf.reduce_max(
                        self.encoder_inputs_length) * 4)

                (
                    self.decoder_outputs_decode,
                    self.final_state,
                    _ # self.decoder_outputs_length_decode
                ) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=self.time_major,
                    # impute_finished=True,	# error occurs
                    maximum_iterations=max_decode_step,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                ))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id:
                    #     [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(
                    #     self.decoder_outputs_decode.rnn_output,
                    #     axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with
                    # the result of the beamsearch decoder
                    # decoder_pred_decode:
                    #     [batch_size, max_time_step, 1] (output_major=False)

                    # self.decoder_pred_decode = tf.expand_dims(
                    #     self.decoder_outputs_decode.sample_id,
                    #     -1
                    # )

                    dod = self.decoder_outputs_decode
                    self.decoder_pred_decode = dod.sample_id

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0))

                else:
                    # Use beam search to approximately
                    # find the most likely translation
                    # decoder_pred_decode:
                    # [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = \
                        self.decoder_outputs_decode.predicted_ids

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0, 2))

                    self.decoder_pred_decode = tf.transpose(
                        self.decoder_pred_decode,
                        perm=[0, 2, 1])
                    dod = self.decoder_outputs_decode
                    self.beam_prob = dod.beam_search_decoder_output.scores


    def save(self, sess, save_path='model.ckpt'):
        """保存模型"""
        self.saver.save(sess, save_path=save_path)


    def load(self, sess, save_path='model.ckpt'):
        """读取模型"""
        print('try load model from', save_path)
        self.saver.restore(sess, save_path)


    def init_optimizer(self):
        """初始化优化器
        支持的方法有 sgd, adadelta, adam, rmsprop, momentum
        """
        # print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        # 'adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

        # 使用包括rewards的loss进行更新
        # 是强化学习的一部分
        gradients = tf.gradients(self.loss_rewards, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.updates_rewards = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)


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


    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length, rewards=None):
        """训练模型"""

        # 输入
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )

        # 设置 dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        if rewards is None:
            # 输出
            output_feed = [self.updates, self.loss]
            _, cost = sess.run(output_feed, input_feed)

            return cost

        else:
            input_feed[self.rewards.name] = rewards
            # output_feed = [self.updates_rewards, self.loss_rewards]
            output_feed = [
                self.updates_rewards,
                self.loss_without_rewards
            ]
            _, cost = sess.run(output_feed, input_feed)

            return cost


    def get_encoder_embedding(self, sess, encoder_inputs):
        """获取经过embedding的encoder_inputs"""

        input_feed = {
            self.encoder_inputs.name: encoder_inputs
        }

        emb = sess.run(self.encoder_inputs_embedded, input_feed)
        return emb


    def entropy(self, sess, encoder_inputs, encoder_inputs_length,
                decoder_inputs, decoder_inputs_length):
        """获取针对一组输入输出的entropy"""
        # 输入
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )

        # 设置 dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.train_entropy, self.decoder_pred_train]

        entropy, logits = sess.run(output_feed, input_feed)
        return entropy, logits


    def predict(self, sess,
                encoder_inputs,
                encoder_inputs_length,
                attention=False):
        """预测输出"""

        # 输入
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            None, None,
            True
        )

        input_feed[self.keep_prob_placeholder.name] = 1.0

        if attention:

            if self.use_beamsearch_decode:

                pred, atten = sess.run([
                    self.decoder_pred_decode,
                    self.final_state[1].alignment_history.stack()
                ], input_feed)

                return pred[:, 0], atten

            pred, atten = sess.run([
                self.decoder_pred_decode,
                self.final_state[1].alignment_history.stack()
            ], input_feed)

            return pred, atten

        # else:

        if self.use_beamsearch_decode:
            pred, beam_prob = sess.run([
                self.decoder_pred_decode, self.beam_prob
            ], input_feed)
            beam_prob = np.mean(beam_prob, axis=1)
            # print(pred.shape, pred)
            # print(beam_prob.shape, beam_prob)

            return pred[:, 0]

        pred, = sess.run([
            self.decoder_pred_decode
        ], input_feed)

        return pred


def transform_data(q, a, ws_q, ws_a, q_max, a_max):
    """转换数据
    """
    x = ws_q.transform(q, max_len=q_max)#, add_end=True)
    y = ws_a.transform(a, max_len=a_max)
    xl = len(q) # q_max# + 1 # len(q)
    yl = len(a) # a_max # len(a)
    # yl = max_len
    return x, xl, y, yl


def batch_flow(x_data, y_data, ws_q, ws_a, batch_size):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    """

    all_data = list(zip(x_data, y_data))

    while True:

        data_batch = random.sample(all_data, batch_size)

        q_max = max([len(x[0]) for x in data_batch])
        a_max = max([len(x[1]) for x in data_batch])
        data_batch = sorted(data_batch, key=lambda x: len(x[1]), reverse=True)

        x_batch = []
        y_batch = []
        xlen_batch = []
        ylen_batch = []

        for q, a in data_batch:
            x, xl, y, yl = transform_data(
                q, a, ws_q, ws_a, q_max, a_max
            )
            x_batch.append(x)
            xlen_batch.append(xl)
            y_batch.append(y)
            ylen_batch.append(yl)

        yield (
            np.array(x_batch),
            np.array(xlen_batch),
            np.array(y_batch),
            np.array(ylen_batch)
        )



def batch_flow_bucket(x_data, y_data, ws_q, ws_a, batch_size, n_bucket=4):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    一个 trick
    相当于把不同数据的根据 target 句子的长度分组，算是一种 bucket
    这里弄的比较简单，复杂一点的是把“相近长度”的输出聚合到一起
    例如输出句子长度1~3的一组，4~6的一组
    每个batch不会出现不同组的长度
    """
    sizes = sorted(list(set([len(y) for y in y_data])))
    buckets = (np.linspace(0, 1, n_bucket + 1) * len(sizes)).astype(int)
    print('buckets', buckets)

    sizes_data = {}
    for i, k in enumerate(buckets):
        if i > 0:
            low = buckets[i - 1]
            v = [(x, y) for x, y in zip(x_data, y_data)
                 if len(y) > low and len(y) < k]
            sizes_data[k] = v
            while len(sizes_data[k]) < batch_size:
                sizes_data[k] = sizes_data[k] + sizes_data[k]
    sizes = sorted(list(sizes_data.keys()))
    # print('sizes', buckets)

    assert tuple(buckets[1:]) == tuple(sizes), \
        '{} != {}'.format(buckets, sizes)
    assert len(sizes) == n_bucket

    while True:

        size = random.choice(sizes)
        data_batch = random.sample(sizes_data[size], batch_size)

        q_max = max([len(x[0]) for x in data_batch])
        a_max = max([len(x[1]) for x in data_batch])
        data_batch = sorted(data_batch, key=lambda x: len(x[1]), reverse=True)

        x_batch = []
        y_batch = []
        xlen_batch = []
        ylen_batch = []

        for q, a in data_batch:
            x, xl, y, yl = transform_data(
                q, a, ws_q, ws_a, q_max, a_max
            )
            x_batch.append(x)
            xlen_batch.append(xl)
            y_batch.append(y)
            ylen_batch.append(yl)

        yield (
            np.array(x_batch),
            np.array(xlen_batch),
            np.array(y_batch),
            np.array(ylen_batch)
        )
