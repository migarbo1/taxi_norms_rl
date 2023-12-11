import pickle
import os

def save_object(path, name, ob):
    with open(os.path.join(path, name+'.pickle'), 'wb') as handler:
        pickle.dump(ob, handler)


def load_object(path, name):
    with open(os.path.join(path, name+'.pickle'), 'rb') as handler:
        return pickle.load(handler)