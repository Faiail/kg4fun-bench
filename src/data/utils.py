from src.utils.strenum import StrEnum


class BatchKeys(StrEnum):
    INPUT_NODE = "input_node"
    SCHEMA_NODE = "schema_node"
    QID = "qid"
    CLS_IDX = "cls_idx"
    ITEM_LABEL = "itemLabel"
    ITEM_DESC = "itemDescription"
    IDX = "idx"
    GT = "gt"
    HEAD_QID = "head_qid"
    HEAD_CLS = "head_cls"
    TAIL_QID = "tail_qid"
    TAIL_CLS = "tail_cls"
    EDGE_TYPE = "edge_type"
    PID = "pid"
    COMPONENT_ID = "component_id"
    INPUT_EDGE = "input_edge"
    SCHEMA_EDGE = "schema_edge"