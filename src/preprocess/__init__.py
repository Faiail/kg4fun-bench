from .ontomap import *

def preprocess_fn(parameters, cls):
    globals()[cls](parameters)()