from torch.utils.data import DataLoader
from src.data.utils import BatchKeys
from .text_complete_dataset import TextCompleteDataset
from typing import Tuple
from .val_text_edge_dataset import SchemaEdgeDataset
from .val_text_node_dataset import SchemaNodeDataset


class ValTextCompleteDataset(TextCompleteDataset):
    def __init__(
        self,
        dataset: str,
        partition_dir: str,
        node_info: str,
        schema_node_info: str,
        input_edge_info: str,
        schema_edge_info: str,
    ):
        super().__init__(
            dataset=dataset,
            partition_dir=partition_dir,
            node_info=node_info,
            schema_node_info=schema_node_info,
            input_edge_info=input_edge_info,
            schema_edge_info=schema_edge_info,
        )
        self.rel_cls2idx_kb = {
            cls_name: idx
            for (idx, cls_name) in enumerate(set(self.dataset.values()) | {-1})
        }
        self.rel_idx2cls_kb = {
            idx: cls_name for (cls_name, idx) in self.rel_cls2idx_kb.items()
        }
        self.node_cls2idx_kb = {
            cls_name: idx
            for (idx, cls_name) in enumerate(list(self.schema_node_info.keys()) + [-1])
        }
        self.node_idx2cls_kb = {
            idx: cls_name for (cls_name, idx) in self.node_cls2idx_kb.items()
        }
        self.node_num_classes = len(list(self.node_cls2idx_kb.keys()))
        self.rel_num_classes = len(self.rel_cls2idx_kb.keys())

    def _get_node_gt(self, cls_idx: int) -> int:
        return self.node_cls2idx_kb[cls_idx]

    def _get_rel_gt(self, head_cls: int, tail_cls: int, pid: str) -> int:
        if head_cls == -1 or tail_cls == -1:
            return self.cls2idx_kb[-1]
        return self.cls2idx_kb[self.dataset[(head_cls, pid, tail_cls)]]

    def _get_gt(self, head_cls: int, tail_cls: int, pid: str) -> int:
        return {
            BatchKeys.HEAD: self._get_node_gt(head_cls),
            BatchKeys.TAIL: self._get_node_gt(tail_cls),
            BatchKeys.REL: self._get_rel_gt(
                head_cls=head_cls, tail_cls=tail_cls, pid=pid
            ),
        }

    def __getitem__(self, index):
        raw_item = self.partitions[index]
        head_qid = raw_item[BatchKeys.HEAD_QID]
        head_cls = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.HEAD_CLS]
        input_head_str = self._get_node_label_desc(head_qid)
        tail_qid = raw_item[BatchKeys.TAIL_QID]
        tail_cls = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.TAIL_CLS]
        input_tail_str = self._get_node_label_desc(tail_qid)
        pid = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.PID]
        input_rel_str = self._get_edge_label_desc(pid)
        gts = self._get_gt(head_cls=head_cls, tail_cls=tail_cls, pid=pid)
        return {
            BatchKeys.HEAD_QID: head_qid,
            BatchKeys.INPUT_HEAD: input_head_str,
            BatchKeys.INPUT_REL: input_rel_str,
            BatchKeys.PID: (
                -1
                if head_cls == -1 or tail_cls == -1
                else self.dataset[(head_cls, pid, tail_cls)]
            ),
            BatchKeys.INPUT_TAIL: input_tail_str,
            BatchKeys.TAIL_QID: tail_qid,
            BatchKeys.GT: gts,
        }

    def get_schema_items(
        self, batch_size: int = 1, num_workers: int = None
    ) -> Tuple[DataLoader, DataLoader]:
        node_dataset = SchemaNodeDataset(
            kb=self.node_idx2cls_kb, schema_info=self.schema_node_info
        )
        node_loader = DataLoader(
            dataset=node_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )
        rel_dataset = SchemaEdgeDataset(
            kb=self.rel_idx2cls_kb, schema_edge_info=self.schema_edge_info
        )
        rel_loader = DataLoader(
            dataset=rel_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )
        return node_loader, rel_loader
