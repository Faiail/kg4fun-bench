from .ontomap import OntomapKG4FUNProcessor

def preprocess_fn(parameters, cls):
    globals()[cls](parameters)()