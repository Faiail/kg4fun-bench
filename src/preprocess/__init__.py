from .ontomap import *
from .inductive_split_processor import InductiveSplitProcessor
from .hybrid_split_processor import HybridSplitProcessor
from .logmap import LogMapPreprocessor
from .schema_converter import SchemaConverter
from .input_kg_converter import InputKGConverter
from .node_feature_extractor import NodeFeatureExtractor


def preprocess_fn(parameters, cls):
    globals()[cls](parameters)()
