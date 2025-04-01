import glob  # Modul za pretragu datoteka u direktorijima prema određenom uzorku
import numpy as np  # NumPy biblioteka za rad s matricama i nizovima podataka
import os.path  # Modul za rad s putanjama datoteka i provjeru postojanja datoteka
import pickle  # Modul za spremanje i učitavanje Python objekata u binarne datoteke
import timeit  # Modul za mjerenje vremena izvršavanja koda

# Definiranje putanja za spremanje učitanih podataka u datoteke
pickle_file = 'data/cache_data.pkl'  # Datoteka za spremanje podataka pomoću pickle-a
numpy_file = 'data/numpy_data.npy'  # Datoteka za spremanje numeričkih podataka u NumPy formatu
books_dir = 'books/'  # Direktorij u kojem se nalaze tekstualne datoteke za obradu

# Parametri za generiranje sekvenci teksta
SEQUENCE_LENGTH = 50  # Duljina svake sekvence znakova koja se koristi za treniranje
STEP = 5  # Korak pomaka pri generiranju sekvenci

def load_from_pickle():
    """Učitava prethodno spremljene podatke iz pickle i numpy datoteka ako postoje."""
    print('Opening from pickle cache')
    with open(pickle_file, "rb") as f:  # Otvaramo pickle datoteku u binarnom načinu čitanja
        data = pickle.load(f)  # Učitavamo spremljeni rječnik podataka
    with open(numpy_file, "rb") as f:  # Otvaramo numpy datoteku u binarnom načinu čitanja
        network_input = np.load(f)  # Učitavamo ulazne podatke
        network_output = np.load(f)  # Učitavamo izlazne podatke
    return network_input, network_output, data["char_indices"], data["indices_char"]

def save_to_pickle(network_input, network_output, char_indices, indices_char):
    """Spremanje podataka u pickle i numpy datoteke radi bržeg budućeg učitavanja."""
    print('Saving to pickle')
    with open(pickle_file, "wb") as f:  # Otvaramo pickle datoteku u binarnom načinu pisanja
        pickle.dump({"char_indices": char_indices, "indices_char": indices_char}, f)  # Spremamo podatke u pickle
    with open(numpy_file, "wb") as f:  # Otvaramo numpy datoteku u binarnom načinu pisanja
        np.save(f, network_input)  # Spremamo ulazne podatke
        np.save(f, network_output)  # Spremamo izlazne podatke

def prepare_data_from_txt(dir, maxlen=SEQUENCE_LENGTH, step=STEP, reloadFresh=False):
    """Učitava tekstualne podatke, generira sekvence i priprema ih za obradu u modelu."""
    # Ako nije postavljeno ponovno učitavanje i postoji pickle datoteka, učitavamo podatke iz nje
    if not reloadFresh and os.path.isfile(pickle_file):
        print('Loading from pickle file...')
        return load_from_pickle()
    
    print('Loading from txt files...')
    text = ""  # Inicijaliziramo prazan string koji će sadržavati sve tekstove
    
    # Pretražujemo sve tekstualne datoteke u direktoriju i spajamo njihov sadržaj
    for file in glob.glob(dir + "**/*.txt", recursive=True):  # Rekurzivno tražimo sve .txt datoteke
        with open(file, 'r', encoding='utf-8') as f:  # Otvaramo svaku datoteku s UTF-8 kodiranjem
            text += f.read()  # Dodajemo sadržaj datoteke u naš glavni tekst
    
    # Kreiramo skup svih jedinstvenih znakova u tekstu i sortiramo ih
    chars = sorted(set(text))
    print(f'Total unique characters: {len(chars)}')  # Ispisujemo broj jedinstvenih znakova
    
    # Stvaramo dvije mape za pretvaranje znakova u brojeve i obrnuto
    char_indices = {c: i for i, c in enumerate(chars)}  # Mapa znak -> indeks
    indices_char = {i: c for i, c in enumerate(chars)}  # Mapa indeks -> znak
    
    network_input = []  # Lista za ulazne sekvence znakova
    network_output = []  # Lista za izlazne znakove (sljedeći znak nakon sekvence)
    
    # Generiranje sekvenci znakova i njihovih ciljanih izlaza
    for i in range(0, len(text) - maxlen, step):  # Iteriramo kroz tekst s korakom 'step'
        seq_in = text[i:i + maxlen]  # Ulazna sekvenca od 'maxlen' znakova
        seq_out = text[i + maxlen]  # Ciljni izlazni znak (znak koji dolazi nakon sekvence)
        network_input.append([char_indices[char] for char in seq_in])  # Pretvaramo znakove u brojeve i spremamo u listu
        network_output.append(char_indices[seq_out])  # Pretvaramo izlazni znak u broj i spremamo ga
    
    # Broj generiranih sekvenci
    n_patterns = len(network_input)
    n_vocab = len(chars)  # Broj različitih znakova
    print(f'Generated {n_patterns} sequences.')  # Ispisujemo broj generiranih sekvenci
    
    # Inicijalizacija matrica za podatke (one-hot encoding)
    X = np.zeros((n_patterns, maxlen, n_vocab), dtype=bool)  # Matrica za ulazne podatke
    y = np.zeros((n_patterns, n_vocab), dtype=bool)  # Matrica za izlazne podatke
    
    # Popunjavanje matrica one-hot vrijednostima
    for i, seq in enumerate(network_input):  # Iteriramo kroz sve sekvence
        for t, char_idx in enumerate(seq):  # Iteriramo kroz svaki znak unutar sekvence
            X[i, t, char_idx] = 1  # Postavljamo 1 na odgovarajuću poziciju u X
        y[i, network_output[i]] = 1  # Postavljamo 1 na odgovarajuću poziciju u y
    
    # Spremamo podatke u datoteke kako bismo ih mogli kasnije koristiti bez ponovnog računanja
    save_to_pickle(X, y, char_indices, indices_char)
    
    return X, y, char_indices, indices_char  # Vraćamo pripremljene podatke

# Mjerenje vremena izvršavanja skripte
start_time = timeit.default_timer()  # Početak mjerenja vremena
network_input, network_output, char_indices, indices_char = prepare_data_from_txt(books_dir)
end_time = timeit.default_timer()  # Kraj mjerenja vremena
print(f"Execution time: {end_time - start_time:.2f} seconds")  # Ispis vremena izvršavanja
