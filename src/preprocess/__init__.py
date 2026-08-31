from .ontomap import *
from .inductive_split_processor import InductiveSplitProcessor

def preprocess_fn(parameters, cls):
    globals()[cls](parameters)()