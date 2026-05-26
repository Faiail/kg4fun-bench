from src.utils.strenum import StrEnum


class OntologyFields(StrEnum):
    NAME = "name"
    IRI = "iri"
    LABEL = "label"
    CHILDRENS = "childrens"
    PARENTS = "parents"
    SYNONYMS = "synonyms"
    COMMENT = "comment"