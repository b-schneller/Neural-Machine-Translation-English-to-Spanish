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
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.args.load_checkpoint is not None:
            self.load(self.args.load_checkpoint)

        source_reversed_lookup_dict = {v: k for k, v in self.data['source_dictionary'].items()}
        target_reversed_lookup_dict = {v: k for k, v in self.data['target_dictionary'].items()}


        bucket_keys = list(self.data['bucket_dictionary'].keys())

        batch_size = self.args.batch_size
        n_epochs = self.args.n_epochs

        loss_tracker = []

        source_eval_sentences, target_eval_sentences = self.get_batch(0, self.data['bucket_dictionary'][5], batch_size,
                                                                      loss_eval=True)
        X_in_eval, y_in_eval, y_target_one_hot_eval = self.get_batch(0, self.data['bucket_dictionary'][5], batch_size)
        eval_feed_dict = {self.X: X_in_eval,
                          self.y_in: y_in_eval,
                          self.y_target_one_hot: y_target_one_hot_eval}

        for epoch in range(n_epochs):
            # shuffle indices in each bucket
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
                                 self.y_target_one_hot: y_out_one_hot_batch}
                    self.sess.run(self.training_op, feed_dict=feed_dict)

                    if iteration % 500 == 0:
                        eval_loss = self.sess.run(self.loss, feed_dict=eval_feed_dict)
                        print('Epoch: %d, Bucket: %d, Iteration: %d/%d, Loss: %f' % (epoch, bucket, iteration,
                                                                                     len(bucket_indices)//batch_size,
                                                                                     eval_loss))
                        eval_output = self.sess.run(self.outputs, feed_dict=eval_feed_dict)
                        eval_output = np.argmax(eval_output, axis=2)
                        loss_tracker.append(eval_loss)

                counter = 0
                num_lines = 30
                for x_in, y_target, output in zip(source_eval_sentences, target_eval_sentences, eval_output):
                    if counter < num_lines:
                        print()
                        print('ENG\t\t',[source_reversed_lookup_dict[word] for word in x_in])
                        print('SPAN_target\t',[target_reversed_lookup_dict[word] for word in y_target])
                        print('SPAN_out\t',[target_reversed_lookup_dict[i] for i in output])
                        print()
                        counter += 1
            self.save(epoch)

    def build_training_graph(self):
        batch_size = self.args.batch_size
        n_inputs = self.args.embedding_size
        n_neurons = self.args.n_neurons
        vocab_size = self.args.vocabulary_size
        n_layers = self.args.n_layers

        self.X = tf.placeholder(tf.float32, [batch_size, None, n_inputs])  #
        self.y_in = tf.placeholder(tf.float32, [batch_size, None, n_inputs])
        self.y_target_one_hot = tf.placeholder(tf.int32, [batch_size, None, vocab_size])
        # self.sequence_length = tf.placeholder(tf.float32, None)
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

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y_target_one_hot)
        self.loss = tf.reduce_mean(self.cross_entropy) #/ self.sequence_length * 10

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.clipped_grads_and_vars = [(tf.clip_by_value(grad, -self.args.max_gradient_norm, self.args.max_gradient_norm),
                                        var) for grad, var in self.grads_and_vars]
        self.training_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars)

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

    def load(self, model_name):
        print(" [*] Loading checkpoint...")
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.args.saved_model_directory, model_name))
