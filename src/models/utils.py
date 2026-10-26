from src.utils.strenum import StrEnum


class ReturnKeys(StrEnum):
    INPUT = "input"
    TARGET = "target"
    HEAD = "head"
    REL = "rel"
    TAIL = "tail"
    TOKEN = "token"
    CLS = "cls"
    MASK = "mask"