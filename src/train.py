import numpy as np
import tensorflow as tf
from datetime import datetime
from random import shuffle
import matplotlib.pyplot as plt
import os


class Train:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.build_training_graph()

    def train(self):
        source_reversed_lookup_dict = {v: k for k, v in self.data['source_dictionary'].items()}
        target_reversed_lookup_dict = {v: k for k, v in self.data['target_dictionary'].items()}
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        bucket_keys = list(self.data['bucket_dictionary'].keys())
        batch_size = self.args.batch_size
        n_epochs = self.args.n_epochs

        loss_tracker = []

        for epoch in range(n_epochs):
            # shuffle indices in each bucket
            source_eval_sentences, target_eval_sentences = self.get_batch(0, self.data['bucket_dictionary'][10], batch_size, loss_eval=True)
            X_in_eval, y_in_eval, y_target_one_hot_eval = self.get_batch(0, self.data['bucket_dictionary'][10], batch_size)
            eval_feed_dict = {self.X: X_in_eval,
                              self.y_in: y_in_eval,
                              self.y_target_one_hot: y_target_one_hot_eval}
            for bucket in bucket_keys:
                shuffle(self.data['bucket_dictionary'][bucket])
            # shuffle buckets
            shuffle(bucket_keys)
            for bucket in bucket_keys:
                bucket_indices = self.data['bucket_dictionary'][bucket]
                for iteration in range(0, len(bucket_indices) // batch_size):
                    X_in_batch, y_in_batch, y_out_one_hot_batch = self.get_batch(iteration, bucket_indices, batch_size)
                    feed_dict = {self.X: X_in_batch,
                                 self.y_in: y_in_batch,
                                 self.y_target_one_hot: y_out_one_hot_batch}#,
                                 # self.sequence_lengths: sequence_lengths}
                    self.sess.run(self.training_op, feed_dict=feed_dict)

                epoch_loss = self.sess.run(self.loss, feed_dict=eval_feed_dict)
                print('Epoch: ', epoch, 'Bucket: ', bucket, ' Total Eval Loss: ', epoch_loss)
                eval_output = self.sess.run(self.outputs, feed_dict=eval_feed_dict)
                eval_output = np.argmax(eval_output, axis=2)
                num_lines = 10
                counter = 0
                for x_in, y_target, output in zip(source_eval_sentences, target_eval_sentences, eval_output):
                    if counter < num_lines:
                        print()
                        print('ENG\t\t',[source_reversed_lookup_dict[word] for word in x_in])
                        print('SPAN_target\t',[target_reversed_lookup_dict[word] for word in y_target])
                        print('SPAN_out\t',[target_reversed_lookup_dict[i] for i in output])
                        print()
                        counter += 1

                loss_tracker.append(epoch_loss)
                plt.plot(loss_tracker)
                plt.xlabel('Epoch')
                plt.ylabel('Eval Loss')
                plt.show()

            self.save(epoch)

    def build_training_graph(self):
        batch_size = self.args.batch_size
        n_inputs = 150
        n_neurons = 64
        vocab_size = 50000
        n_layers = 3

        self.X = tf.placeholder(tf.float32, [batch_size, None, n_inputs])  #
        self.y_in = tf.placeholder(tf.float32, [batch_size, None, n_inputs])
        self.y_target_one_hot = tf.placeholder(tf.int32, [batch_size, None, vocab_size])
        # self.sequence_lengths = tf.placeholder(tf.int32, [batch_size, None])
        # self.target_weights = tf.placeholder(tf.float32, [None])


       # Encoder
        self.layers_encode = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.tanh)
                  for layer in range(n_layers)]

        self.multi_layer_cell_encode = tf.contrib.rnn.MultiRNNCell(self.layers_encode)

        _, self.state = tf.nn.dynamic_rnn(self.multi_layer_cell_encode, self.X, dtype=tf.float32)

        # Decoder
        self.layers_decode = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.tanh)
                              for layer in range(n_layers)]

        self.multi_layer_cell_decode = tf.contrib.rnn.MultiRNNCell(self.layers_decode)

        self.output_cell = tf.contrib.rnn.OutputProjectionWrapper(self.multi_layer_cell_decode, output_size=vocab_size)

        self.outputs, _ = tf.nn.dynamic_rnn(self.output_cell, self.y_in, initial_state=self.state, dtype=tf.float32)

        learning_rate = self.args.learning_rate

        # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y_target_one_hot)
        # self.truncated_cross_entropy = np.multiply(self.cross_entropy, self.sequence_lengths)
        # self.loss = tf.reduce_mean(self.truncated_cross_entropy)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y_target_one_hot)
        self.loss = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

    def get_batch(self, iteration, indices, batch_size, loss_eval=False): # < ----- Needs work
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
        if loss_eval:
            return X_in_batch_indices, y_out_batch_indices

        else:
            return X_in_batch, y_in_batch, y_out_one_hot

    def save(self, epoch):
        print('[*] Saving checkpoint ....')
        model_name = 'nmt_model_epoch_{}.ckpt'.format(epoch)
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, os.path.join(self.args.saved_model_directory, model_name))
        print('[*] Checkpoint saved in file {}'.format(save_path))

    def load(self):
        pass
