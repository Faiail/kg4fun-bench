from src.utils.strenum import StrEnum


class BatchKeys(StrEnum):
    INPUT = "input"
    SCHEMA = "schema"
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
    EDGE_TYPES = "edge_types"
    PID = "pid"
    COMPONENT_ID = "component_id"