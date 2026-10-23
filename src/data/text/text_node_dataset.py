from torch.utils.data import Dataset
from src.data.utils import BatchKeys
from src.utils import load_json
from .template_keys import TemplateKeys
import torch


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

    def get_schema_str(self, idx: int) -> str:
        if idx == -1:
            return f"{TemplateKeys.LABEL} Prune Node | {TemplateKeys.DESC} A node which is out of domain and must be pruned."
        return f"{TemplateKeys.LABEL} {self.schema_info[idx].get(BatchKeys.ITEM_LABEL, "UnknownLabel")} | {TemplateKeys.DESC} {self.schema_info[idx].get(BatchKeys.ITEM_DESC, "UnknownDescription")}"

    def __getitem__(self, idx: int) -> dict:
        raw_data = self.dataset[idx]
        input_qid = raw_data[BatchKeys.QID]
        target_class = raw_data[BatchKeys.CLS_IDX]

        input_node_str = f"{TemplateKeys.LABEL} {self.node_info[input_qid].get(BatchKeys.ITEM_LABEL, "UnknownLabel")} | {TemplateKeys.DESC} {self.node_info[input_qid].get(BatchKeys.ITEM_DESC, "UnknownDescription")}"

        return {
            BatchKeys.INPUT: input_node_str,
            BatchKeys.SCHEMA: target_class,
        }

    def collate_fn(self, batch: list) -> dict:
        input_nodes_str = [item[BatchKeys.INPUT] for item in batch]
        target_classes = [item[BatchKeys.SCHEMA] for item in batch]
        unique_classes = list(set(target_classes))
        class2batch = {v: k for k, v in enumerate(unique_classes)}

        target_classes_str = [self.get_schema_str(idx) for idx in unique_classes]
        gts = torch.as_tensor([class2batch[idx] for idx in target_classes], dtype=torch.long)
        return {
            BatchKeys.INPUT: input_nodes_str,
            BatchKeys.SCHEMA: target_classes_str,
            BatchKeys.GT: gts,
        }