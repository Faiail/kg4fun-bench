from torch.utils.data import Dataset, DataLoader
from src.data.utils import BatchKeys
from .template_keys import TemplateKeys
from .text_edge_dataset import TextEdgeDataset


class ValTextEdgeDataset(TextEdgeDataset):
    def __init__(
        self,
        dataset: str,
        partition_dir: str,
        node_info: str,
        input_edge_info: str,
        schema_edge_info: str,
    ):
        super().__init__(
            dataset=dataset,
            partition_dir=partition_dir,
            node_info=node_info,
            input_edge_info=input_edge_info,
            schema_edge_info=schema_edge_info,
        )
        self.cls2idx_kb = {
            cls_name: idx
            for (idx, cls_name) in enumerate(list(self.dataset.keys()) + [-1, -1, -1])
        }
        self.idx2cls_kb = {idx: cls_name for (cls_name, idx) in self.cls2idx_kb.items()}

    def __getitem__(self, index):
        raw_item = self.partitions[index]
        head_qid = raw_item[BatchKeys.HEAD_QID]
        head_cls = raw_item[BatchKeys.HEAD_CLS]
        head_str = self._get_node_label_desc(head_qid)
        tail_qid = raw_item[BatchKeys.TAIL_QID]
        tail_cls = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.TAIL_CLS]
        tail_str = self._get_node_label_desc(tail_qid)
        pid = raw_item[BatchKeys.EDGE_TYPE][BatchKeys.PID]
        rel_str = self._get_edge_label_desc(pid)
        input_edge_str = f"{TemplateKeys.HEAD} {head_str} | {TemplateKeys.TAIL} {tail_str} | {TemplateKeys.PID} {rel_str}"
        gt = self._get_gt(head_cls=head_cls, tail_cls=tail_cls, pid=pid)
        return {
            BatchKeys.INPUT: input_edge_str,
            BatchKeys.GT: gt,
        }

    def _get_gt(self, head_cls: int, tail_cls: int, pid: str) -> int:
        if head_cls == -1 or tail_cls == -1:
            return self.cls2idx_kb[[-1, -1, -1]]
        return self.cls2idx_kb[[head_cls, pid, tail_cls]]

    def get_schema_rels(
        self, batch_size: int = 1, num_workers: int = None
    ) -> DataLoader:
        dataset = SchemaEdgeDataset(
            dataset=self.idx2cls_kb, schema_edge_info=self.schema_edge_info
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )


class SchemaEdgeDataset(Dataset):
    def __init__(self, dataset: dict, schema_edge_info: dict) -> None:
        super().__init__()
        self.dataset = dataset
        self.schema_edge_info = schema_edge_info
        self.l = list(self.dataset.keys())

    def __len__(self):
        return len(self.l)

    def _get_schema_rel(self, id: int) -> str:
        pass

    def __getitem__(self, index):
        raw_id = self.dataset[self.l[index]]
        schema_rel = self._get_schema_rel(self, raw_id)
