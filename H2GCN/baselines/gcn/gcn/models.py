from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        # self.activations.append(self.inputs)
        # for layer in self.layers:
        #     hidden = layer(self.activations[-1])
        #     self.activations.append(hidden)
        # self.outputs = self.activations[-1]

        # Build skip connection layer model 1: add skip connection both on mlp and graphnorm
        # self.activations.append(self.inputs)
        # for i in range(FLAGS.dense_layers):
        #     hidden = self.layers[i](self.activations[-1])
        #     self.activations.append(hidden)
        # self.activations.append(tf.math.add_n(
        #     [self.activations[1], self.activations[-1]]))
        # for i in range(FLAGS.dense_layers, FLAGS.dense_layers + FLAGS.norm_layers):
        #     hidden = self.layers[i](self.activations[-1])
        #     self.activations.append(hidden)
        # self.activations.append(tf.math.add_n(
        #     [self.activations[FLAGS.dense_layers+1], self.activations[-1]]))
        # self.outputs = self.layers[-1](self.activations[-1])

        # Build skip connection layer model 2: only add skip connection on graphnorm
        self.activations.append(self.inputs)
        for i in range(FLAGS.dense_layers):
            hidden = self.layers[i](self.activations[-1])
            self.activations.append(hidden)

        H0 = self.activations[-1]
        for i in range(FLAGS.dense_layers, FLAGS.dense_layers + FLAGS.norm_layers):
            hidden = self.layers[i]((self.activations[-1], H0))
            self.activations.append(hidden)
        self.outputs = tf.math.add_n([H0, self.activations[-1]])

        # Build jump connection layer model
        # self.activations.append(self.inputs)
        # layers_num = len(self.layers)
        # for i in range(layers_num - 1):
        #     hidden = self.layers[i](self.activations[-1])
        #     self.activations.append(hidden)
        # norm_sum = self.activations[layers_num-FLAGS.norm_layers]
        # for j in range(FLAGS.norm_layers - 1):
        #     norm_sum = tf.concat(
        #         [norm_sum, self.activations[layers_num-FLAGS.norm_layers+j+1]], axis=1)
        # self.outputs = self.layers[-1](norm_sum)

        # Store model variables for easy access
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # self.layers.append(GroupNorm(input_dim=FLAGS.hidden1,
        #                              output_dim=FLAGS.hidden1,
        #                              alpha=FLAGS.alpha,
        #                              beta=FLAGS.beta,
        #                              placeholders=self.placeholders,
        #                              # act=tf.nn.relu,
        #                              # dropout=True,
        #                              # sparse_inputs=True,
        #                              logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCNConcat2(GCN):
    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.activations.append(tf.concat(self.activations[-3:], axis=-1))
        with tf.variable_scope(self.name):
            classifier = Dense(input_dim=3*FLAGS.hidden1,
                               output_dim=self.output_dim,
                               placeholders=self.placeholders,
                               act=lambda x: x,
                               dropout=True,
                               logging=self.logging)
        self.layers.append(classifier)
        self.activations.append(classifier(self.activations[-1]))
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))


# class MLP_NORM(Model):
#     def __init__(self, placeholders, input_dim, **kwargs):
#         super(MLP_NORM, self).__init__(**kwargs)

#         self.inputs = placeholders['features']
#         self.input_dim = input_dim
#         # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
#         self.output_dim = placeholders['labels'].get_shape().as_list()[1]
#         self.placeholders = placeholders

#         self.optimizer = tf.train.AdamOptimizer(
#             learning_rate=FLAGS.learning_rate)

#         self.build()

#     def _loss(self):
#         # Weight decay loss
#         for var in self.layers[0].vars.values():
#             self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

#         # Cross entropy error
#         self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
#                                                   self.placeholders['labels_mask'])

#     def _accuracy(self):
#         self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
#                                         self.placeholders['labels_mask'])

#     def _build(self):
#         self.layers.append(Dense(input_dim=self.input_dim,
#                                  output_dim=FLAGS.hidden1,
#                                  placeholders=self.placeholders,
#                                  act=tf.nn.relu,
#                                  dropout=False,
#                                  sparse_inputs=True,
#                                  logging=self.logging))

#         self.layers.append(Dense(input_dim=FLAGS.hidden1,
#                                  output_dim=self.output_dim,
#                                  placeholders=self.placeholders,
#                                  act=lambda x: x,
#                                  dropout=True,
#                                  logging=self.logging))
#         for i in range(FLAGS.norm_layers):
#             self.layers.append(GroupNorm(input_dim=self.output_dim,
#                                          output_dim=self.output_dim,
#                                          alpha=FLAGS.alpha,
#                                          beta=FLAGS.beta,
#                                          placeholders=self.placeholders,
#                                          # act=tf.nn.relu,
#                                          # dropout=True,
#                                          # sparse_inputs=True,
#                                          logging=self.logging))
#         self.layers.append(Dense(input_dim=self.output_dim * FLAGS.norm_layers,
#                                  output_dim=self.output_dim,
#                                  placeholders=self.placeholders,
#                                  act=lambda x: x,
#                                  dropout=True,
#                                  logging=self.logging))

#     def predict(self):
#         return tf.nn.softmax(self.outputs)


class MLP_NORM(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP_NORM, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=False,
                                 sparse_inputs=True,
                                 logging=self.logging))
        for _ in range(FLAGS.dense_layers-2):
            self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     dropout=True,
                                     logging=self.logging))
        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))
        for _ in range(FLAGS.norm_layers):
            self.layers.append(GroupNorm(input_dim=self.output_dim,
                                         output_dim=self.output_dim,
                                         alpha=FLAGS.alpha,
                                         beta=FLAGS.beta,
                                         gamma=FLAGS.gamma,
                                         placeholders=self.placeholders,
                                         logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
