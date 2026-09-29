from torch.utils.data import Dataset
from src.data.utils import BatchKeys
from src.utils import load_json
from .template_keys import TemplateKeys


class TextNodeDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        node_info: str,
        schema_info: str,
    ) -> None:
        super().__init__()
        self.dataset = load_json(dataset)
        self.node_info = load_json(node_info)
        self.schema_info = load_json(schema_info)
        self.schema_info = {item.get(BatchKeys.IDX): item for item in self.schema_info}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        raw_data = self.dataset[idx]
        input_qid = raw_data[BatchKeys.QID]
        target_class = raw_data[BatchKeys.CLS_IDX]

        input_node_str = f"{TemplateKeys.LABEL} {self.node_info[input_qid].get(BatchKeys.ITEM_LABEL, "UnknownLabel")} | {TemplateKeys.DESC} {self.node_info[input_qid].get(BatchKeys.ITEM_DESC, "UnknownDescription")}"

        if target_class == -1:
            target_class_str = f"{TemplateKeys.LABEL}PruneNode | {TemplateKeys.DESC} A node which is out of domain and must be pruned."
        else:
            target_class_str = f"{TemplateKeys.LABEL} {self.schema_info[target_class].get(BatchKeys.ITEM_LABEL, "UnknownLabel")} | {TemplateKeys.DESC} {self.schema_info[target_class].get(BatchKeys.ITEM_DESC, "UnknownDescription")}"

        return {
            BatchKeys.INPUT_NODE: input_node_str,
            BatchKeys.SCHEMA_NODE: target_class_str,
        }
