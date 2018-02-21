import numpy as np
import tensorflow as tf
from random import shuffle
import os

class Train:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.build_training_graph()
        self.source_reversed_lookup_dict = {v: k for k, v in self.data['source_dictionary'].items()}
        self.target_reversed_lookup_dict = {v: k for k, v in self.data['target_dictionary'].items()}

    def train(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            if self.args.load_checkpoint is not None:
                self.load(self.args.load_checkpoint)

            bucket_keys = list(self.data['bucket_dictionary'].keys())

            batch_size = self.args.batch_size
            n_epochs = self.args.n_epochs

            loss_tracker = []

            source_eval_sentences, target_eval_sentences = self.get_batch(0, self.data['bucket_dictionary'][15], batch_size,
                                                                          loss_eval=True)
            X_in_eval, y_in_eval, y_target_one_hot_eval = self.get_batch(0, self.data['bucket_dictionary'][15], batch_size)
            eval_feed_dict = {self.X: X_in_eval,
                              self.y_in: y_in_eval,
                              self.y_target_one_hot: y_target_one_hot_eval,
                              self.inference_bool: False,
                              self.infer_state: np.zeros((2, 2, 80, 128))}

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
                                     self.y_target_one_hot: y_out_one_hot_batch,
                                     self.inference_bool: False,
                                     self.infer_state: np.zeros((2,2,80,128))}
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
                            print('ENG\t\t', ' '.join([self.source_reversed_lookup_dict[word] for word in x_in]))
                            print('SPAN_target\t', ' '.join([self.target_reversed_lookup_dict[word] for word in y_target]))
                            print('SPAN_out\t', ' '.join([self.target_reversed_lookup_dict[i] for i in output]))
                            print()
                            counter += 1
                self.save(epoch)

    def infer(self):
        input_sentence = self.args.input_sentence
        numerical_id_sentence = [self.data['source_dictionary'][i] if i in self.data['source_dictionary']
                                 else self.data['source_dictionary']['<UNK>'] for i in input_sentence.split()][::-1]
        source_input = np.array([self.data['source_embeddings'][i] for i in numerical_id_sentence])
        translation = list()
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            if self.args.load_checkpoint is not None:
                self.load(self.args.load_checkpoint)
            else:
                print('Inference requires loading a trained model.')


            translation_input = self.data['target_embeddings'][self.data['target_dictionary']['<GO>']]

            encoding_state = self.sess.run(self.state, feed_dict={self.X: source_input[np.newaxis, :]})

            while True:
                next_word, next_state = self.sess.run([self.outputs, self.output_state], feed_dict={self.X: source_input[np.newaxis, :],
                                                                                                    self.y_in: translation_input[np.newaxis, np.newaxis, :],
                                                                                                    self.inference_bool: True,
                                                                                                    self.infer_state: encoding_state})
                translation.append(np.argmax(next_word))

                translation_input = self.data['target_embeddings'][np.argmax(next_word)]
                encoding_state = next_state
                if np.argmax(next_word) == self.data['target_dictionary']['<EOS>']:
                    print(' '.join([self.target_reversed_lookup_dict[i] for i in translation]))
                    break
                print('nw:', np.argmax(next_word, axis=2))

    def build_training_graph(self):
        batch_size = self.args.batch_size
        n_inputs = self.args.embedding_size
        n_neurons = self.args.n_neurons
        vocab_size = self.args.vocabulary_size
        n_layers = self.args.n_layers

        self.X = tf.placeholder(tf.float32, [batch_size, None, n_inputs])  #
        self.y_in = tf.placeholder(tf.float32, [batch_size, None, n_inputs])
        self.y_target_one_hot = tf.placeholder(tf.int32, [batch_size, None, vocab_size])
        self.inference_bool = tf.placeholder(tf.bool)
        self.infer_state = tf.placeholder(tf.float32, [2, 2, batch_size, n_neurons])

        l = tf.unstack(self.infer_state, axis=0)
        self.inference_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(n_layers)])

        # Encoder
        self.layers_encode = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.tanh)
                  for layer in range(n_layers)]

        self.multi_layer_cell_encode = tf.contrib.rnn.MultiRNNCell(self.layers_encode, state_is_tuple=True)

        _, self.state = tf.nn.dynamic_rnn(self.multi_layer_cell_encode, self.X, dtype=tf.float32)


        # Decoder
        self.layers_decode = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.tanh)
                              for layer in range(n_layers)]

        self.multi_layer_cell_decode = tf.contrib.rnn.MultiRNNCell(self.layers_decode, state_is_tuple=True)

        self.output_cell = tf.contrib.rnn.OutputProjectionWrapper(self.multi_layer_cell_decode, output_size=vocab_size)

        self.outputs, self.output_state = tf.cond(self.inference_bool,
                                                  lambda: tf.nn.dynamic_rnn(self.output_cell, self.y_in, initial_state=self.inference_state, dtype=tf.float32),
                                                  lambda: tf.nn.dynamic_rnn(self.output_cell, self.y_in, initial_state=self.state, dtype=tf.float32))

        # self.outputs, self.output_state = tf.nn.dynamic_rnn(self.output_cell, self.y_in, initial_state=self.state, dtype=tf.float32)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y_target_one_hot)
        self.loss = tf.reduce_mean(self.cross_entropy) #/ self.sequence_length * 10

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.clipped_grads_and_vars = [(tf.clip_by_value(grad, -self.args.max_gradient_norm, self.args.max_gradient_norm),
                                        var) for grad, var in self.grads_and_vars]
        self.training_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars)

    def get_batch(self, iteration, indices, batch_size, loss_eval=False):
        begin = iteration * batch_size
        end = (iteration + 1) * batch_size
        extraction_indices = indices[begin:end]

        X_in_batch_indices = [self.data['X_in'][i][::-1] for i in extraction_indices]
        X_in_batch = np.array([self.data['source_embeddings'][i] for i in X_in_batch_indices])

        y_in_batch_indices = [self.data['y_in'][i] for i in extraction_indices]
        y_in_batch = np.array([self.data['target_embeddings'][i] for i in y_in_batch_indices])

        y_out_batch_indices = [self.data['y_out'][i] for i in extraction_indices]

        y_out_one_hot = np.zeros((batch_size, X_in_batch.shape[1], 50000))
        for row, sentence in enumerate(y_out_batch_indices):
            for col, word in enumerate(sentence):
                y_out_one_hot[row, col, word] = 1

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
