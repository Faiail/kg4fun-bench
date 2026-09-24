from src.utils.strenum import StrEnum


class ClassFields(StrEnum):
    LABEL = "itemLabel"
    DESCRIPTION = "itemDescription"
    IDX = "idx"

class RelFields(StrEnum):
    PID = "pid"
    LABEL = "label"
    HEAD_CLS = "head_cls"
    TAIL_CLS = "tail_cls"
    TARGET = "target"
    ITEM_LABEL = "itemLabel"
    ITEM_DESCRIPTION = "itemDescription"