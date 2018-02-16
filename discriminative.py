

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


class Discriminative(object):

    def __init__(self,
                 input_vocab_size, batch_size,
                 mode='train', depth=1, embedding_size=128,
                 hidden_units=64, cell_type='lstm',
                 bidirectional=False,
                 use_residual=False, use_dropout=False,
                 dropout=0.1, time_major=True,
                 parallel_iterations=32,
                 optimizer='adam',
                 learning_rate=0.001,
                 max_gradient_norm=5.0,
                 seed=0):
        self.input_vocab_size = input_vocab_size
        self.batch_size = batch_size
        self.mode = mode
        self.depth = depth
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.cell_type = cell_type
        self.use_residual = use_residual
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.time_major = time_major
        self.bidirectional = bidirectional
        self.parallel_iterations = parallel_iterations
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.keep_prob = 1.0 - dropout
        self.seed = seed

        # Initialize encoder_embeddings to have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        self.initializer = tf.random_uniform_initializer(
            -sqrt3, sqrt3, dtype=tf.float32
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

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name='keep_prob'
        )


        self.init_placeholders()
        self.build_encoder()
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
            self.targets = tf.placeholder(
                dtype=tf.float32,
                shape=(self.batch_size, 2),
                name='target'
            )


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

            inputs = self.encoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs, (1, 0, 2))

            if not self.bidirectional:
                (
                    self.encoder_outputs,
                    _
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
                    (_, _)
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


            # self.encoder_outputs
            if self.time_major:
                self.encoder_outputs = tf.transpose(
                    self.encoder_outputs, (1, 0, 2))

            self.encoder_outputs = self.encoder_outputs[:, -1, :]

            hidden_units = self.hidden_units
            if self.bidirectional:
                hidden_units *= 2

            softmax_w = tf.get_variable(
                name='softmax_w',
                shape=(hidden_units, 2),
                initializer=self.initializer,
                dtype=tf.float32
            )
            softmax_b = tf.get_variable(
                name='softmax_b',
                shape=(2,),
                initializer=tf.zeros_initializer(),
                dtype=tf.float32
            )

            self.outputs = tf.nn.softmax(tf.nn.xw_plus_b(
                self.encoder_outputs,
                softmax_w, softmax_b, name='softmax_output'))


            if self.mode == 'train':
                self.loss = tf.reduce_mean((self.outputs - self.targets) ** 2)


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



    def predict(self, sess, encoder_inputs, encoder_inputs_length):

        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.keep_prob_placeholder.name: 1.0
        }

        output_feed = self.outputs

        return sess.run(output_feed, input_feed)


    def train(self, sess, encoder_inputs, encoder_inputs_length, targets):

        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.targets: targets,
            self.keep_prob_placeholder.name: self.keep_prob
        }

        output_feed = [self.updates, self.loss, self.outputs]

        return sess.run(output_feed, input_feed)[1:]


    def save(self, sess, save_path='model.ckpt'):
        """保存模型"""
        self.saver.save(sess, save_path=save_path)


    def load(self, sess, save_path='model.ckpt'):
        """读取模型"""
        print('try load model from', save_path)
        self.saver.restore(sess, save_path)

def test():
    """单元测试"""
    from fake_data import generate
    from data_utils import batch_flow
    from tqdm import tqdm

    x_data, y_data, ws_input, ws_target = generate(size=10000)

    batch_size = 4
    n_epoch = 10
    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:

            model = Discriminative(len(ws_input), batch_size=batch_size)
            init = tf.global_variables_initializer()
            sess.run(init)

            # print(sess.run(model.input_layer.kernel))
            # exit(1)

            for epoch in range(1, n_epoch + 1):
                costs = []
                flow = batch_flow(
                    x_data, y_data, ws_input, ws_target, batch_size
                )
                bar = tqdm(range(steps),
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    targets = np.array([
                        (0, 1)
                        for x in range(len(y))
                    ])
                    cost = model.train(sess, x, xl, targets)
                    print(x.shape, xl.shape)
                    print('cost.shape, cost', cost.shape, cost)
                    exit(1)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f}'.format(
                        epoch,
                        np.mean(costs)
                    ))


if __name__ == '__main__':
    test()
