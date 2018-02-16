import numpy as np
import tensorflow as tf
from random import shuffle


class Train:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.build_training_graph()

    def train(self):

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        bucket_keys = list(self.data['bucket_dictionary'].keys())
        batch_size = 50
        n_epochs = self.args.n_epochs

        for epoch in range(n_epochs):
            for bucket in bucket_keys:
                bucket_indices = self.data['bucket_dictionary'][bucket]
                for iteration in range(0, len(bucket_indices) // batch_size):
                    print(iteration)
                    X_in_batch, y_in_batch, y_out_one_hot_batch = self.get_batch(iteration, bucket_indices, batch_size)
                    feed_dict = {self.X: X_in_batch,
                                 self.y_in: y_in_batch,
                                 self.y_target_one_hot: y_out_one_hot_batch}#,
                                 # self.sequence_lengths: sequence_lengths}
                    self.sess.run(self.training_op, feed_dict=feed_dict)

    def build_training_graph(self):
        batch_size = self.args.batch_size
        n_inputs = 150  # embedding vector length
        n_neurons = 128  # whatever
        vocab_size = 50000  # vocab size and length of one-hot vector

        self.X = tf.placeholder(tf.float32, [batch_size, None, n_inputs])  #
        self.y_in = tf.placeholder(tf.float32, [batch_size, None, n_inputs])
        self.y_target_one_hot = tf.placeholder(tf.int32, [batch_size, None, vocab_size])
        # self.sequence_lengths = tf.placeholder(tf.int32, [batch_size, None])
        # self.target_weights = tf.placeholder(tf.float32, [None])

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)

        # Encoder
        _, self.state = tf.nn.dynamic_rnn(self.lstm_cell, self.X, dtype=tf.float32)

        # Decoder
        self.output_cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.LSTMCell(num_units=n_neurons),
                                                             output_size=vocab_size)

        self.outputs, _ = tf.nn.dynamic_rnn(self.output_cell, self.y_in, dtype=tf.float32)

        learning_rate = 0.001

        # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y_target_one_hot)
        # self.truncated_cross_entropy = np.multiply(self.cross_entropy, self.sequence_lengths)
        # self.loss = tf.reduce_mean(self.truncated_cross_entropy)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y_target_one_hot)
        self.loss = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

    def get_batch(self, iteration, indices, batch_size): # < ----- Needs work
        begin = iteration * batch_size
        end = (iteration + 1) * batch_size
        extraction_indices = indices[begin:end]   # good to here

        X_in_batch_indices = [self.data['X_in'][i] for i in extraction_indices]    #list of lists of num_id sentences
        X_in_batch = np.array([self.data['source_embeddings'][i] for i in X_in_batch_indices])

        y_in_batch_indices = [self.data['y_in'][i] for i in extraction_indices]
        y_in_batch = np.array([self.data['target_embeddings'][i] for i in y_in_batch_indices])

        y_out_batch_indices = [self.data['y_out'][i] for i in extraction_indices]

        y_out_one_hot = np.zeros((batch_size, X_in_batch.shape[1], 50000))
        for row, sentence in enumerate(y_out_batch_indices):
            for col, word in enumerate(sentence):
                y_out_one_hot[row, col, word] = 1


        # sequence_lengths = np.array(y_out_batch_indices) != self.data['target_dictionary']['<PAD>']

        return X_in_batch, y_in_batch, y_out_one_hot

    def save(self):
        pass

    def load(self):
        pass
