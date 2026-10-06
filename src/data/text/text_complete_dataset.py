from .text_edge_dataset import TextEdgeDataset
from .template_keys import TemplateKeys
from src.data.utils import BatchKeys
from src.utils import load_json
import torch


class TextCompleteDataset(TextEdgeDataset):
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
            input_edge_info=input_edge_info,
            schema_edge_info=schema_edge_info,
        )
        self.schema_node_info = load_json(schema_node_info)
        self.schema_node_info = {
            item.get(BatchKeys.IDX): item for item in self.schema_node_info
        }

    def _get_schema_node_label_desc(self, cls_idx: int) -> str:
        if cls_idx == -1:
            return f"{TemplateKeys.LABEL} Prune Node | {TemplateKeys.DESC} A node which is out of domain and must be pruned."
        return f"{TemplateKeys.LABEL} {self.schema_node_info[cls_idx].get(BatchKeys.ITEM_LABEL, 'Unknown Label')} | {TemplateKeys.DESC} {self.schema_node_info[cls_idx].get(BatchKeys.ITEM_DESC, 'Unknown Description')}"

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

        return {
            BatchKeys.INPUT_HEAD: input_head_str,
            BatchKeys.INPUT_REL: input_rel_str,
            BatchKeys.INPUT_TAIL: input_tail_str,
            BatchKeys.TARGET_HEAD: head_cls,
            BatchKeys.TARGET_REL: tuple((head_cls, pid, tail_cls)),
            BatchKeys.TARGET_TAIL: tail_cls,
        }

    def collate_fn(self, batch: list) -> dict:
        input_heads_str = [item[BatchKeys.INPUT_HEAD] for item in batch]
        input_rels_str = [item[BatchKeys.INPUT_REL] for item in batch]
        input_tails_str = [item[BatchKeys.INPUT_TAIL] for item in batch]
        target_heads = [item[BatchKeys.TARGET_HEAD] for item in batch]
        target_rels = [item[BatchKeys.TARGET_REL] for item in batch]
        target_tails = [item[BatchKeys.TARGET_TAIL] for item in batch]

        unique_head_classes = list(set(target_heads))
        unique_tail_classes = list(set(target_tails))
        unique_rel_classes = list(set(target_rels))

        head_class2batch = {v: k for k, v in enumerate(unique_head_classes)}
        tail_class2batch = {v: k for k, v in enumerate(unique_tail_classes)}
        rel_class2batch = {v: k for k, v in enumerate(unique_rel_classes)}

        target_heads_str = [
            self._get_schema_node_label_desc(cls_idx) for cls_idx in unique_head_classes
        ]
        target_tails_str = [
            self._get_schema_node_label_desc(cls_idx) for cls_idx in unique_tail_classes
        ]
        target_rels_str = [
            self._get_schema_rel_label_desc(head_cls=head, tail_cls=tail, pid=pid)
            for (head, pid, tail) in unique_rel_classes
        ]

        gts_heads = torch.as_tensor(
            [head_class2batch[cls_idx] for cls_idx in target_heads], dtype=torch.long
        )
        gts_tails = torch.as_tensor(
            [tail_class2batch[cls_idx] for cls_idx in target_tails], dtype=torch.long
        )
        gts_rels = torch.as_tensor(
            [rel_class2batch[rep] for rep in target_rels], dtype=torch.long
        )

        return {
            BatchKeys.INPUT_HEAD: input_heads_str,
            BatchKeys.INPUT_REL: input_rels_str,
            BatchKeys.INPUT_TAIL: input_tails_str,
            BatchKeys.TARGET_HEAD: target_heads_str,
            BatchKeys.TARGET_REL: target_rels_str,
            BatchKeys.TARGET_TAIL: target_tails_str,
            BatchKeys.GT: {
                BatchKeys.HEAD: gts_heads,
                BatchKeys.REL: gts_rels,
                BatchKeys.TAIL: gts_tails,
            },
        }
