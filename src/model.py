import numpy as np
import tensorflow as tf
from random import shuffle
import os


class NMT_Model:
    def __init__(self, args, data):
        self.args = args
        self.bucket_dictionary = data['bucket_dictionary']
        self.source_dictionary = data['source_dictionary']
        self.target_dictionary = data['target_dictionary']
        self.source_embeddings = data['source_embeddings']
        self.target_embeddings = data['target_embeddings']
        self.source_input = data['source_input']
        self.target_input = data['target_input']
        self.target_output = data['target_output']
        self.source_reversed_lookup_dict = data['source_reverse_dictionary']
        self.target_reversed_lookup_dict = data['target_reverse_dictionary']
        self.build_training_graph()

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

        with tf.name_scope('Encoder'):
            self.layers_encode = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.tanh)
                      for layer in range(n_layers)]

            self.multi_layer_cell_encode = tf.contrib.rnn.MultiRNNCell(self.layers_encode, state_is_tuple=True)

            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.multi_layer_cell_encode, self.X, dtype=tf.float32)

        with tf.name_scop('Decocer'):
            self.layers_decode = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.tanh)
                                  for layer in range(n_layers)]

            self.multi_layer_cell_decode = tf.contrib.rnn.MultiRNNCell(self.layers_decode, state_is_tuple=True)


            self.output_cell = tf.contrib.rnn.OutputProjectionWrapper(self.multi_layer_cell_decode, output_size=vocab_size)

            self.decoder_output, self.output_state = tf.cond(self.inference_bool,
                                                      lambda: tf.nn.dynamic_rnn(self.output_cell, self.y_in, initial_state=self.inference_state, dtype=tf.float32),
                                                      lambda: tf.nn.dynamic_rnn(self.output_cell, self.y_in, initial_state=self.encoder_state, dtype=tf.float32))

            self.softmax_output = tf.nn.softmax(self.decoder_output) # needed for inference

        with tf.name_scope('Loss'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.decoder_output, labels=self.y_target_one_hot)
            self.loss = tf.reduce_mean(self.cross_entropy)

        with tf.name_scope('Train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.clipped_grads_and_vars = [(tf.clip_by_value(grad, -self.args.max_gradient_norm, self.args.max_gradient_norm),
                                            var) for grad, var in self.grads_and_vars]
            self.training_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars)

    def train(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())

            if self.args.load_checkpoint is not None:
                self.load(self.args.load_checkpoint)

            bucket_keys = list(self.bucket_dictionary.keys())
            batch_size = self.args.batch_size
            n_epochs = self.args.n_epochs
            dummy_infer_state = np.zeros((self.args.n_layers, 2, self.args.batch_size, self.args.n_neurons))

            for epoch in range(n_epochs):
                shuffle(bucket_keys)
                for bucket in bucket_keys:
                    shuffle(self.bucket_dictionary[bucket])
                for bucket in bucket_keys:
                    bucket_indices = self.bucket_dictionary[bucket]
                    for iteration in range(0, len(bucket_indices) // batch_size):
                        source_input_batch, target_input_batch, target_output_one_hot_batch = self.get_batch(iteration, bucket_indices, batch_size)

                        feed_dict = {self.X: source_input_batch,
                                     self.y_in: target_input_batch,
                                     self.y_target_one_hot: target_output_one_hot_batch,
                                     self.inference_bool: False,
                                     self.infer_state: dummy_infer_state}

                        self.sess.run(self.training_op, feed_dict=feed_dict)

                        if iteration % 500 == 0:
                            self.evaluate(epoch, bucket, iteration, bucket_indices)

                self.save(epoch)

    def evaluate(self, epoch, bucket, iteration, bucket_indices):
        batch_size = self.args.batch_size

        X_in_eval, y_in_eval, y_target_one_hot_eval = self.get_batch(0, self.bucket_dictionary[15], batch_size)
        eval_feed_dict = {self.X: X_in_eval,
                          self.y_in: y_in_eval,
                          self.y_target_one_hot: y_target_one_hot_eval,
                          self.inference_bool: False,
                          self.infer_state: np.zeros((self.args.n_layers, 2, self.args.batch_size, self.args.n_neurons))}

        eval_loss = self.sess.run(self.loss, feed_dict=eval_feed_dict)
        print('Epoch: %d, Bucket: %d, Iteration: %d/%d, Loss: %f' % (epoch, bucket, iteration,
                                                                     len(bucket_indices) // batch_size,
                                                                     eval_loss))

    def infer(self):
        input_sentence = self.args.input_sentence
        numerical_id_sentence = [self.source_dictionary[i] if i in self.args.input_sentence
                                 else self.source_dictionary['<UNK>'] for i in input_sentence.split()][::-1]
        source_input = np.array([self.source_embeddings[i] for i in numerical_id_sentence])
        translation = list()

        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            if self.args.load_checkpoint is not None:
                self.load(self.args.load_checkpoint)
            else:
                print('Inference requires loading a trained model.')

            translation_input = self.target_embeddings[self.target_dictionary['<GO>']]
            encoding_state = self.sess.run(self.encoder_state, feed_dict={self.X: source_input[np.newaxis, :]})
            translation = list()

            while True:
                next_word, next_state = self.sess.run([self.decoder_output, self.output_state],
                                                      feed_dict={self.X: source_input[np.newaxis, :],
                                                                 self.y_in: translation_input[np.newaxis, np.newaxis,:],
                                                                 self.inference_bool: True,
                                                                 self.infer_state: encoding_state})

                translation.append(np.argmax(next_word))
                translation_input = self.target_embeddings[np.argmax(next_word)]
                encoding_state = next_state
                if np.argmax(next_word) == self.target_dictionary['<EOS>']:
                    print(' '.join([self.target_reversed_lookup_dict[i] for i in translation]))
                    break

    def beam_search(self, next_word):
        pass

    def get_batch(self, iteration, indices, batch_size, loss_eval=False):
        begin = iteration * batch_size
        end = (iteration + 1) * batch_size
        extraction_indices = indices[begin:end]

        X_in_batch_indices = [self.source_input[i][::-1] for i in extraction_indices]
        X_in_batch = np.array([self.source_embeddings[i] for i in X_in_batch_indices])

        y_in_batch_indices = [self.target_input[i] for i in extraction_indices]
        y_in_batch = np.array([self.target_embeddings[i] for i in y_in_batch_indices])

        y_out_batch_indices = [self.target_output[i] for i in extraction_indices]

        y_out_one_hot = np.zeros((batch_size, X_in_batch.shape[1], self.args.vocabulary_size), dtype=np.int32)
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
