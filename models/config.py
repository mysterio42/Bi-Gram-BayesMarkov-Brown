import random
import string
import glob
import os
import joblib

MODEL_DIR = 'bigram/'


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def dump_embedding(embedding):
    path = MODEL_DIR + 'markov-bigram-' + generate_model_name(5) + '.pkl'

    with open(path, 'wb') as f:
        joblib.dump(value=embedding, filename=f, compress=3)
        print(f'Embedding saved at {path}')

def latest_modified_embedding():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    embedding_files = glob.glob(MODEL_DIR + '*')
    latest = max(embedding_files, key=os.path.getctime)
    return latest


def load_embedding():

    path = latest_modified_embedding()

    with open(path, 'rb') as f:
        return joblib.load(filename=f)
