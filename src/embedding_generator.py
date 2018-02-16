import random
from collections import deque
import tensorflow as tf
import numpy as np


class Embedding_model:
    def __init__(self, args):
        self.args = args
        self.build_embedding_graph()

    def build_embedding_graph(self):

        vocabulary_size = self.args.vocabulary_size
        embedding_size = self.args.embedding_size

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        num_sampled = 64  # Number of negative examples to sample.
        batch_size = 128

        learning_rate = 0.01

        # Input data.
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        self.valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        self.init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        self.embeddings = tf.Variable(self.init_embeds)

        self.train_inputs = tf.placeholder(tf.int32, shape=[None])
        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

        # Construct the variables for the NCE loss
        self.nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / np.sqrt(embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.train_labels, self.embed,
                                             num_sampled, vocabulary_size))

        # Construct the Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), axis=1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / self.norm
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

    def train(self, numerical_id):
        num_steps = 50001
        data_index = 0

        numerical_id = [word for sentence in numerical_id for word in sentence]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('[*] Beginning training...')

        average_loss = 0

        for step in range(num_steps):
            print("\rIteration: {}".format(step), end="\t")
            batch_inputs, batch_labels, data_index = self.generate_batch(numerical_id, data_index)
            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            # We perform one update step by evaluating the training op (including it
            # in the list of returned values for session.run()
            _, loss_val = self.sess.run([self.training_op, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 10000 == 0:
                if step > 0:
                    average_loss /= 10000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

        embeddings = self.sess.run(self.normalized_embeddings)
        return embeddings

    def generate_batch(self, numerical_id, data_index):
        # borrowed from https://github.com/ageron/handson-ml/blob/master/14_recurrent_neural_networks.ipynb
        skip_window = 1
        num_skips = 2
        batch_size = 128
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = deque(maxlen=span)
        for _ in range(span):
            buffer.append(numerical_id[data_index])
            data_index = (data_index + 1) % len(numerical_id)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(numerical_id[data_index])
            data_index = (data_index + 1) % len(numerical_id)
        return batch, labels, data_index
