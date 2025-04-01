import glob
import numpy as np
import os.path
import pickle
import timeit

pickle_file = 'data/cache_data.pkl'
numpy_file = 'data/numpy_data.npy'
books_dir = 'books/'

SEQUENCE_LENGTH = 50
STEP = 5

def load_from_pickle():
    print('Opening from pickle cache')
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    with open(numpy_file, "rb") as f:
        network_input = np.load(f)
        network_output = np.load(f)
    return network_input, network_output, data["char_indices"], data["indices_char"]

def save_to_pickle(network_input, network_output, char_indices, indices_char):
    print('Saving to pickle')
    with open(pickle_file, "wb") as f:
        pickle.dump({"char_indices": char_indices, "indices_char": indices_char}, f)
    with open(numpy_file, "wb") as f:
        np.save(f, network_input)
        np.save(f, network_output)

def prepare_data_from_txt(dir, maxlen=SEQUENCE_LENGTH, step=STEP, reloadFresh=False):
    if not reloadFresh and os.path.isfile(pickle_file):
        print('Loading from pickle file...')
        return load_from_pickle()
    
    print('Loading from txt files...')
    text = ""
    for file in glob.glob(dir + "**/*.txt", recursive=True):
        with open(file, 'r', encoding='utf-8') as f:
            text += f.read()
    
    chars = sorted(set(text))
    print(f'Total unique characters: {len(chars)}')
    
    char_indices = {c: i for i, c in enumerate(chars)}
    indices_char = {i: c for i, c in enumerate(chars)}
    
    network_input = []
    network_output = []
    
    for i in range(0, len(text) - maxlen, step):
        seq_in = text[i:i + maxlen]
        seq_out = text[i + maxlen]
        network_input.append([char_indices[char] for char in seq_in])
        network_output.append(char_indices[seq_out])
    
    n_patterns = len(network_input)
    n_vocab = len(chars)
    print(f'Generated {n_patterns} sequences.')
    
    X = np.zeros((n_patterns, maxlen, n_vocab), dtype=bool)
    y = np.zeros((n_patterns, n_vocab), dtype=bool)
    
    for i, seq in enumerate(network_input):
        for t, char_idx in enumerate(seq):
            X[i, t, char_idx] = 1
        y[i, network_output[i]] = 1
    
    save_to_pickle(X, y, char_indices, indices_char)
    return X, y, char_indices, indices_char

# Mjerenje vremena
start_time = timeit.default_timer()
network_input, network_output, char_indices, indices_char = prepare_data_from_txt(books_dir, SEQUENCE_LENGTH, STEP)
end_time = timeit.default_timer()
print(f"Execution time: {end_time - start_time:.2f} seconds")
