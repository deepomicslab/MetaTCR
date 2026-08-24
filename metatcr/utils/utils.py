import pickle
import os

def save_pk(file_savepath, data):
    with open(file_savepath, "wb") as fp:
        pickle.dump(data, fp)

def load_pkfile(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    return data

