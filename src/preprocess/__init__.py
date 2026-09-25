from .ontomap import *
from .inductive_split_processor import InductiveSplitProcessor
from .hybrid_split_processor import HybridSplitProcessor
from .logmap import LogMapPreprocessor

def preprocess_fn(parameters, cls):
    globals()[cls](parameters)()