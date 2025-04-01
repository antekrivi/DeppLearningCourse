import glob
import numpy as np
import os.path
import re
import timeit
import torch
import pickle

pickle_file = 'data/cache_data.pkl'
numpy_file = 'data/numpy_data.npy'

SEQUENCE_LENGTH = 50
STEP = 5

def load_from_pickle():
    print('Opening from pickle cache')
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    with open(numpy_file, "rb") as f:
        network_input = np.load(f)
        network_output = np.load(f)
        network_input_labels = np.load(f)
    return network_input, network_output, network_input_labels, data[0], data[1], data[2], data[3], data[4]


def save_to_pickle(network_input, network_output, network_input_labels, words, indices_char, char_indices, label_indices, indices_label):
    print('Saving to pickle')
    with open(pickle_file, "wb") as f:
        pickle.dump([words, indices_char, char_indices, label_indices, indices_label], f)
    with open(numpy_file, "wb") as f:
        np.save(f, network_input)
        np.save(f, network_output)
        np.save(f, network_input_labels)


def prepare_data_from_csv(dir, maxlen, word_size = 1, step=3, vectorization=True, reloadFresh=False):
    if not reloadFresh and os.path.isfile(pickle_file):
        print('Loading from pickle file...')
        return load_from_pickle()

    print('Loading from txt files...')
    input_data = []
    next_chars = []
    tree_labels = []
    svi = []
    dic = {}

    #trees = []
    for file in glob.glob(dir + "**/*.txt"):
        label = file.split('/')[-2] #uzimamo zadnji folder kao label
        if label not in dic:
            dic[label] = []
        with open(file, 'r') as f:
            temp = f.read()
            visak = len(temp) % word_size
            if visak > 0:
                temp = temp[:-visak]
            dic[label].append(temp)
            svi.append(temp)

    svi = ''.join(svi)

    lista = list(map(''.join, zip(*[iter(svi)] * word_size))) #dijelim listu po veličini riječi
    words = sorted(set(lista))

    print('total words:', len(words))
    char_indices = dict((c, i) for i, c in enumerate(words))
    indices_char = dict((i, c) for i, c in enumerate(words))

    label_indices = dict((c, i) for i, c in enumerate(dic.keys()))
    indices_label = dict((i, c) for i, c in enumerate(dic.keys()))


    network_input = None
    network_output = None

    if vectorization: # radimo vektorizaciju po mapi zato što nam trebaju klase

        #prvo presložimo mapu kako bi u svakoj imali string
        for svaki in dic.keys():
            print("Converting label", svaki)
            dic[svaki] = ''.join(dic[svaki])
            current_label = dic[svaki]

            for i in range(0, len(current_label) - maxlen * word_size, step * word_size):
                input_data.append(current_label[i: i + (maxlen * word_size)])
                next_chars.append(current_label[i + (maxlen * word_size): i + (maxlen * word_size) + word_size])
                tree_labels.append(label_indices[svaki])
        print('nb sequences:', len(input_data))

        print('Vectorization...')
        network_input = np.zeros((len(input_data), maxlen, len(words)), dtype=np.bool)
        network_output = np.zeros((len(input_data), len(words)), dtype=np.bool)
        network_input_labels = np.zeros((len(input_data), len(label_indices)), dtype=np.bool)

        for i, item in enumerate(input_data):
            for t, char in enumerate2(item, start=0, step=word_size):
                network_input[i, int(t / word_size), char_indices[char]] = 1
            network_output[i, char_indices[next_chars[i]]] = 1
            network_input_labels[i, tree_labels[i]] = 1
            if i % 10000 == 0:
                print('Vectorized', i, 'of', len(input_data))

    save_to_pickle(network_input, network_output, network_input_labels, words, indices_char, char_indices, label_indices, indices_label)

    return network_input, network_output, network_input_labels, words, indices_char, char_indices, label_indices, indices_label


def enumerate2(xs, start=0, step=1):
    for i in range(0, len(xs), step):
        yield (start, xs[i:i + step])
        start += step


# Mjerenje vremena
start_time = timeit.default_timer()
network_input, networku_output, network_input_labels, words, indices_char, char_indices, label_indices, indices_label = prepare_data_from_csv('txt/', SEQUENCE_LENGTH, word_size=1, step=STEP, vectorization=True)
end_time = timeit.default_timer()
print(f"Execution time: {end_time - start_time:.2f} seconds")