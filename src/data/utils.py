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