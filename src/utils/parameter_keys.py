from .strenum import StrEnum


class ParameterKeys(StrEnum):
    GENERAL = "general"
    OUT_DIR = "out_dir"
    PBAR = "pbar"
    DATA = "data"
    NAME = "name"
    CLASS = "class"
    REL = "rel"
    NODE_TYPES = "node_types"
    EDGE_TYPES = "edge_types"
    EDGE_TYPE_INFO = "edge_type_info"
    FILTERED = "filtered"