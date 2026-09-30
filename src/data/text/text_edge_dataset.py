from torch.utils.data import Dataset
from src.utils import load_json
from .template_keys import TemplateKeys
from src.data.utils import BatchKeys
import os


class TextEdgeDataset(Dataset):
    def __init__(
        self,
        dataset: str,  # connected components mapping
        partition_dir: str,  # all partitions
        node_info: str,  # node info
        input_edge_info: str,  # edge info
        schema_edge_info: str,  # edge type info (qwen)
    ):
        super().__init__()
        self.dataset = load_json(dataset)
        self.dataset = {
            tuple(x[BatchKeys.EDGE_TYPE]): x[BatchKeys.COMPONENT_ID] for x in self.dataset
        }
        self.partitions = sum(
            [load_json(f"{partition_dir}/{fname}") for fname in os.listdir(partition_dir)], start=list()
        )
        self.node_info = load_json(node_info)
        self.input_edge_info = load_json(input_edge_info)
        self.schema_edge_info = load_json(schema_edge_info)
        self.schema_edge_info = {
            x[BatchKeys.IDX]: x for x in self.schema_edge_info
        }

    def __len__(self):
        return len(self.partitions)

    def __getitem__(self, index):
        raw_item = self.partitions[index]
        head_qid = raw_item[BatchKeys.HEAD_QID]
        head_cls = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.HEAD_CLS]
        head_str = self._get_node_label_desc(head_qid)
        tail_qid = raw_item[BatchKeys.TAIL_QID]
        tail_cls = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.TAIL_CLS]
        tail_str = self._get_node_label_desc(tail_qid)
        pid = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.PID]
        rel_str = self._get_edge_label_desc(pid)
        target_edge_str = self._get_schema_rel_label_desc(
            head_cls=head_cls,
            tail_cls=tail_cls,
            pid=pid,
        )
        input_edge_str = f"{TemplateKeys.HEAD} {head_str} | {TemplateKeys.TAIL} {tail_str} | {TemplateKeys.PID} {rel_str}"
        return {
            BatchKeys.INPUT: input_edge_str,
            BatchKeys.SCHEMA: target_edge_str,
        }

    def _get_node_label_desc(self, qid: str) -> str:
        return f"{TemplateKeys.LABEL} {self.node_info.get(qid).get(BatchKeys.ITEM_LABEL, "Unknown Label")} | {TemplateKeys.DESC} {self.node_info.get(qid).get(BatchKeys.ITEM_DESC, "Unknown Description")}"

    def _get_edge_label_desc(self, pid: str) -> str:
        return f"{TemplateKeys.LABEL} {self.input_edge_info.get(pid).get(BatchKeys.ITEM_LABEL, "Uknown Label")} | {TemplateKeys.DESC} {self.input_edge_info.get(pid).get(BatchKeys.ITEM_DESC, "Unknown Description")}"

    def _get_schema_rel_label_desc(self, head_cls: int, tail_cls: int, pid: str) -> str:
        if head_cls == -1 or tail_cls == -1:
            return f"{TemplateKeys.LABEL} Prune Edge | {TemplateKeys.DESC} An edge which is out of domain and must be pruned."
        schema_rel_id = self.dataset[(head_cls, pid, tail_cls)]
        return f"{TemplateKeys.LABEL} {self.schema_edge_info.get(schema_rel_id).get(BatchKeys.ITEM_LABEL, "Unknown Label")} | {TemplateKeys.DESC} {self.schema_edge_info.get(schema_rel_id).get(BatchKeys.ITEM_DESC, "Unknown Description")}"
