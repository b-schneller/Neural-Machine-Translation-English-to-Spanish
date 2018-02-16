import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from collections import Counter
import string
import math


def process_data(args):
    source_fname = '../data/raw/europarl-v7.es-en.en'
    target_fname = '../data/raw/europarl-v7.es-en.es'
    source_sentences = read_sentences_from_file(source_fname)
    target_sentences = read_sentences_from_file(target_fname)
    source_clean, target_clean = clean_sentence_lists(source_sentences, target_sentences)
    source_dictionary, source_vocabulary = build_vocabulary(source_clean)
    target_dictionary, target_vocabulary = build_vocabulary(target_clean)
    bucket_dict = create_bucket_dict(source_clean, target_clean)
    data = add_tokens_to_text(source_clean, target_clean, bucket_dict, source_dictionary, target_dictionary)
    data['source_vocabulary'] = source_vocabulary
    data['source_dictionary'] = source_dictionary
    data['target_vocabulary'] = target_vocabulary
    data['target_dictionary'] = target_dictionary
    data['bucket_dictionary'] = bucket_dict
    return data


def read_sentences_from_file(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
        content = [line.strip('\n') for line in content]
    return content

def clean_sentence_lists(source_list, target_list, max_len=64):
    source_clean, target_clean = list(), list()
    punctuation_translator = str.maketrans('', '' ,string.punctuation)
    punctuation_translator[191] = None  # to remove inverted question mark
    for source, target in zip(source_list, target_list):
        if len(source.split()) < (max_len-1) and len(target.split()) < (max_len-1):
            if source is not '' and target is not '':
                source = source.translate(punctuation_translator)
                source = source.replace(" s ", "'s ")
                target = target.translate(punctuation_translator)
                target = target.replace(" s ", "'s ")
                source_clean.append(source.lower())
                target_clean.append(target.lower())
    return source_clean, target_clean


def build_vocabulary(sentence_list, vocabulary_size=50000):
    tokens = [('<UNK>', None), ('<PAD>', None), ('<EOS>', None), ('<GO>', None)]
    vocabulary_size -= len(tokens)
    word_list = [word for line in sentence_list for word in line.split()]
    vocabulary = tokens + Counter(word_list).most_common(vocabulary_size)
    vocabulary = np.array([word for word, _ in vocabulary])
    dictionary = {word: code for code, word in enumerate(vocabulary)}
    return dictionary, vocabulary


def create_bucket_dict(eng_sentences, span_sentences):
    sample_bucket_sizes = []
    bucket_dict = {}
    for eng_sentence, span_sentence in zip(eng_sentences, span_sentences):
        max_len = max(len(eng_sentence.split()), len(span_sentence.split()))
        rounded_max_len = roundup(max_len)
        sample_bucket_sizes.append(rounded_max_len)
    for i in range(10, max(sample_bucket_sizes) + 1, 10):
        bucket_dict[i] = create_buckets(sample_bucket_sizes, i)

    return bucket_dict


def roundup(x):
    return int(math.ceil((x + 1) / 10.0)) * 10  # x+1 to push *0 into next bucket to account for tokens


def create_buckets(buckets, bucket_len):
    return [index for index, value in enumerate(buckets) if value == bucket_len]


def add_tokens_to_text(source_list, target_list, bucket_dict, source_dictionary, target_dictionary):
    number_of_samples = len(source_list)
    source_final, target_input_final, target_output_final = [None] * number_of_samples, [None] * number_of_samples, [
        None] * number_of_samples
    inverse_bucket_dict = invert(bucket_dict)
    for index, bucket_size in inverse_bucket_dict.items():
        source_final[index] = pad_source_sentences(source_list[index], bucket_size)
        target_input_final[index] = pad_target_input_sentences(target_list[index], bucket_size)
        target_output_final[index] = pad_target_output_sentences(target_list[index], bucket_size)
    source_final_numerical = convert_words_to_numerical_id(source_final, source_dictionary)
    target_input_final_numerical = convert_words_to_numerical_id(target_input_final, target_dictionary)
    target_output_final_numerical = convert_words_to_numerical_id(target_output_final, target_dictionary)

    data = {'X_in': source_final_numerical, 'y_in': target_input_final_numerical, 'y_out': target_output_final_numerical}
    return data


def pad_source_sentences(sentence, bucket_size):
    sentence_length = len(sentence.split())
    pad_length = bucket_size - sentence_length
    return sentence + ' <PAD>' * pad_length


def pad_target_input_sentences(sentence, bucket_size):
    sentence_length = len(sentence.split())
    pad_length = bucket_size - sentence_length - 1
    return '<GO> ' + sentence + ' <PAD>' * pad_length


def pad_target_output_sentences(sentence, bucket_size):
    sentence_length = len(sentence.split())
    pad_length = bucket_size - sentence_length - 1
    return sentence + ' <EOS> ' + ' <PAD>' * pad_length


def invert(dictionary):
    return dict((value, key) for key in dictionary for value in dictionary[key])


def convert_words_to_numerical_id(sentence_list, dictionary):
    out = []
    for sentence in sentence_list:
        out.append([dictionary[word] if word in dictionary else dictionary['<UNK>'] for word in sentence.split()])
    return out
