from torch.utils.data import Dataset, DataLoader
import torch
from src.data.utils import BatchKeys
from .template_keys import TemplateKeys
from .text_node_dataset import TextNodeDataset


class TextNodeValDataset(TextNodeDataset):
    def __init__(self, dataset: str, node_info: str, schema_info: str) -> None:
        super().__init__(dataset=dataset, node_info=node_info, schema_info=schema_info)
        self.cls2idx_kb = {
            cls_name: idx for (idx, cls_name) in enumerate(list(self.schema_info.keys()) + [-1])
        }
        self.idx2cls_kb = {idx: cls_name for (cls_name, idx) in self.cls2idx_kb.items()}
        self.num_classes = len(list(self.cls2idx_kb.keys()))

    def __getitem__(self, idx):
        raw_data = self.dataset[idx]
        input_qid = raw_data[BatchKeys.QID]
        target_class = raw_data[BatchKeys.CLS_IDX]

        input_node_str = f"{TemplateKeys.LABEL}{self.node_info[input_qid].get(BatchKeys.ITEM_LABEL, "UnknownLabel")} | {TemplateKeys.DESC}{self.node_info[input_qid].get(BatchKeys.ITEM_DESC, "UnknownDescription")}"

        gt = self.cls2idx_kb[target_class]

        return {
            BatchKeys.QID: input_qid,
            BatchKeys.INPUT_NODE: input_node_str,
            BatchKeys.GT: gt,
        }

    def get_schema_nodes(
        self, batch_size: int = 1, num_workers: int = None
    ) -> DataLoader:
        dataset = SchemaNodeDataset(
            kb=self.idx2cls_kb,
            schema_info=self.schema_info,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )


class SchemaNodeDataset(Dataset):
    def __init__(self, kb: dict, schema_info: dict) -> None:
        self.schema_info = schema_info
        self.kb = kb

    def __len__(self):
        return len(self.kb)

    def __getitem__(self, index):
        target_class = self.kb[index]
        if target_class == -1:
            target_class_str = f"{TemplateKeys.LABEL}PruneNode | {TemplateKeys.DESC} A node which is out of domain and must be pruned."
        else:
            raw = self.schema_info[target_class]
            target_class_str = f"{TemplateKeys.LABEL} {raw.get(BatchKeys.ITEM_LABEL, "UnknownLabel")} | {TemplateKeys.DESC} {raw.get(BatchKeys.ITEM_DESC, "UnknownDescription")}"
        return {BatchKeys.SCHEMA_NODE: target_class_str}
