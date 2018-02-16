from data_preprocessing import process_data
from make_dataset import make_dataset
from embedding_generator import Embedding_model
from train import Train
import argparse
import sys
import numpy as np
import pickle
import os

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', action='store_true', default=False,
                        help='Turn on to download data to disk.')
    parser.add_argument('--process_data', action='store_true', default=False,
                        help='Turn on to process raw data.')

    parser.add_argument('--raw_data_directory', default='../data/raw/',
                        help='Directory for downloaded language corpus')
    parser.add_argument('--processed_data_directory', default='../data/processed/',
                        help='Directory for processed data (embeddings, language_dicts, etc.')

    parser.add_argument('--generate_embeddings', action='store_true', default=False,
                        help='Turn on to generate word embeddings')
    parser.add_argument('--embedding_size', type=int, default=150,
                        help='Dimensionality of embedding vectors')
    parser.add_argument('--vocabulary_size', type=int, default=50000,
                        help='Number of unique words in vocabulary')

    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for training and evaluation.')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='Number of training epochs')

    parser.add_argument('--saved_model_directory', default='../models/',
                        help='Directory for saving trained models')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Optimizer learning rate')
    parser.add_argument('--early_stopping_max_checks', type=int, default=20,
                        help='Max checks without improvement for early stopping')

    parser.add_argument('--train', action='store_true', default=False,
                        help='Set to True to train network')

    parser.add_argument('--infer', action='store_true', default=False,
                        help='Set to True to conduct inference on Test images. Trained model must be loaded.')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Load saved checkpoint, arg=checkpoint_name')

    args = parser.parse_args()

    os.makedirs(args.raw_data_directory, exist_ok=True)
    os.makedirs(args.processed_data_directory, exist_ok=True)
    os.makedirs(args.saved_model_directory, exist_ok=True)

    if args.download_data:
        make_dataset(args)

    if args.process_data:
        data = process_data(args)

    if args.generate_embeddings:
        embed_model = Embedding_model(args)
        data['source_embeddings'] = embed_model.train(data['X_in'])
        data['target_embeddings'] = embed_model.train(data['y_in'])
        with open('data_processed.pickle', 'wb') as f:
            pickle.dump(data, f)

    if args.train:
        with open('data_processed.pickle', 'rb') as f:
            data = pickle.load(f)
        training_model = Train(args, data)
        training_model.train()


if __name__ == '__main__':
    main(sys.argv)
