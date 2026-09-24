from .ontomap import *
from .inductive_split_processor import InductiveSplitProcessor
from .hybrid_split_processor import HybridSplitProcessor

def preprocess_fn(parameters, cls):
    globals()[cls](parameters)()