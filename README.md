# Neural Machine Translation - English to Spanish

## Introduction

This code implements a sequence-to-sequence recurrent neural network (RNN) for translating English sentences and phrases into Spanish. The network consists of a double layer of 512 unit Long Short Term Memory (LSTM) cells. The training corpus comes from the [European Parliament Proceedings Parallel Corpus 1996-2011](http://www.statmt.org/europarl/). This dataset contains over two million corresponding English and Spanish sentences. 

## To Begin

Clone the repo, move to the source directory, `Neural-Machine-Translation-English-to-Spanish/src/`, and run the command below. This will download the corpus to a local folder, clean and prepare the English and Spanish sentences, generate embeddings for the vocabulary and a set of tokens (`<UNK>, <EOS>, <PAD>, <GO>`), and pickel the prepared datasets for future training and inference.

`python main.py --download_data --process_data --generate_embeddings`

### Optional arguments:

`--embedding_size`: Dimension of the embedding vectors

`--vocabulary_size`: Number of unique words for each language to use. Will select the top most commonly occuring words for the vocabulary. Other words will be replaced by an unknown word token `<UNK>`



## To Train

`python main.py --train`

# Optional settings include:

`--batch_size`: Batch size for training and evaluation

`--n_epochs`: Number of passes through the entire training set

`--n_layers`: Number of layers for the encoder/decoder RNNs

`--n_neurons`: Number of neurons in each RNN layer

`--learning_rate`: Optimizer learning rate

`--max_gradient_norm`: Gradient norm threshold for gradient clipping

`--early_stopping_max_checks`: Max number of checks without improvement for early stopping

## To Generate Translations

`python main.py --infer --load_checkpoint <checkpoint_filename>.ckpt --input_sentence <english input sentence>`

## Sample Translations

Coming soon.

