from .shared_text_single_item_run import SharedTextSingleItemRun
from .utils import ParameterKeys
from src.data.utils import BatchKeys
import torch
from typing import Tuple
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
import src.metrics as metrics_pkg
from copy import deepcopy


class SharedTextCompleteRun(SharedTextSingleItemRun):
    def _init_metric_collection(self, collection_parameters: dict) -> MetricCollection:
        return MetricCollection(
            {
                k: metrics_pkg.__dict__[v.get(ParameterKeys.NAME)](
                    **v.get(ParameterKeys.CFG, {})
                )
                for k, v in collection_parameters.items()
            }
        )

    def _init_metrics(self) -> None:
        metric_parameters = self.parameters.get(ParameterKeys.METRICS, dict())
        node_metric_parameters = metric_parameters.get(ParameterKeys.NODES, dict())
        self.node_metrics = self._init_metric_collection(node_metric_parameters)
        edge_metric_parameters = metric_parameters.get(ParameterKeys.EDGES, dict())
        self.edge_metrics = self._init_metric_collection(edge_metric_parameters)

    def rebuild_metrics(self, num_node_classes: int, num_rel_classes: int) -> None:
        metric_parameters = deepcopy(self.parameters.get(ParameterKeys.METRICS, dict()))
        node_metric_parameters = metric_parameters.get(ParameterKeys.NODES, dict())
        for metric in node_metric_parameters:
            if (
                "num_classes"
                in node_metric_parameters[metric].get(ParameterKeys.CFG, dict()).keys()
            ):
                node_metric_parameters[metric][ParameterKeys.CFG][
                    "num_classes"
                ] = num_node_classes
        self.node_metrics = self._init_metric_collection(node_metric_parameters)
        edge_metric_parameters = metric_parameters.get(ParameterKeys.EDGES, dict())
        for metric in edge_metric_parameters:
            if (
                "num_classes"
                in edge_metric_parameters[metric].get(ParameterKeys.CFG, dict()).keys()
            ):
                edge_metric_parameters[metric][ParameterKeys.CFG][
                    "num_classes"
                ] = num_rel_classes
        self.edge_metrics = self._init_metric_collection(edge_metric_parameters)

    def train_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAIN} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.train()
        self.model_warmup(epoch=epoch)
        for ix, batch in bar:
            self.optimizer.zero_grad()
            input_head_ref = batch[BatchKeys.INPUT_HEAD]
            input_tail_ref = batch[BatchKeys.INPUT_TAIL]
            input_rel_ref = batch[BatchKeys.INPUT_REL]

            target_head_ref = batch[BatchKeys.TARGET_HEAD]
            target_tail_ref = batch[BatchKeys.TARGET_TAIL]
            target_rel_ref = batch[BatchKeys.TARGET_REL]

            input_head_tok, input_tail_tok, input_rel_tok = (
                self.tokenize(input_head_ref),
                self.tokenize(input_tail_ref),
                self.tokenize(input_rel_ref),
            )
            target_head_tok, target_tail_tok, target_rel_tok = (
                self.tokenize(target_head_ref),
                self.tokenize(target_tail_ref),
                self.tokenize(target_rel_ref),
            )
            input_emb, target_emb = self.model(
                input_head=input_head_tok,
                input_tail=input_tail_tok,
                input_rel=input_rel_tok,
                target_head=target_head_tok,
                target_tail=target_tail_tok,
                target_rel=target_rel_tok,
            )
            labels = batch.get(
                BatchKeys.GT,
                {k: torch.arange(v.size(0)).long() for k, v in target_emb.items()},
            )
            labels = {k: v.to(self.device) for k, v in labels.items()}
            loss = self.criterion(input_emb, target_emb, labels)

            loss.backward()
            self.optimizer.step()

            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar=bar, loss=batch_loss)
            self.schedule(phase=ParameterKeys.TRAIN)
            torch.cuda.empty_cache()

        cumulated_loss /= len(self.train_loader)
        self.print_stats(
            epoch=epoch, cumulated_loss=cumulated_loss, phase=ParameterKeys.TRAIN
        )

    @torch.no_grad()
    def get_spec_schema_embeddings(self, loader: DataLoader, desc: str) -> torch.Tensor:
        schema_item_embeddings = torch.empty(
            size=(len(loader.dataset), self.model.schema_hidden_size)
        )
        bar = self.get_bar(loader=loader, desc=desc)
        batch_size = loader.batch_size
        for ix, batch in bar:
            schema_ref = batch[BatchKeys.SCHEMA]
            schema_tokens = self.tokenize(schema_ref)
            schema_embedds = self.model.encode(schema_tokens)
            schema_item_embeddings[(ix * batch_size) : ((ix + 1) * batch_size)] = (
                schema_embedds.cpu()
            )
        return schema_item_embeddings

    @torch.no_grad()
    def get_schema_embeddings(
        self, node_loader: DataLoader, edge_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_spec_schema_embeddings(
            loader=node_loader, desc="Get node classes item embeddings"
        ), self.get_spec_schema_embeddings(
            loader=edge_loader, desc="Get edge rels item embeddings"
        )

    def print_stats(
        self,
        epoch: int,
        cumulated_loss: float,
        phase: str,
        node_metrics: dict = None,
        edge_metrics: dict = None,
    ) -> None:
        if cumulated_loss is not None:
            print(
                f"Epoch {epoch}/{self.num_epochs}: {phase} loss: {cumulated_loss:.4f}"
            )
        if node_metrics:
            for k, v in node_metrics.items():
                print(f"Epoch {epoch}/{self.num_epochs}: {phase} node {k}: {v:.4f}")
        if edge_metrics:
            for k, v in edge_metrics.items():
                print(f"Epoch {epoch}/{self.num_epochs}: {phase} edge {k}: {v:.4f}")

    @torch.no_grad()
    def val_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        self.rebuild_metrics(
            num_node_classes=self.val_loader.dataset.node_num_classes,
            num_rel_classes=self.val_loader.dataset.rel_num_classes,
        )
        self.model.eval()

        node_loader, edge_loader = self.val_loader.dataset.get_schema_items(
            batch_size=self.val_loader.batch_size,
            num_workers=self.val_loader.num_workers,
        )
        schema_node_embeddings, schema_edge_embeddings = self.get_schema_embeddings(
            node_loader=node_loader, edge_loader=edge_loader
        )

        for ix, batch in bar:
            gt = batch.pop(BatchKeys.GT)
            input_head_ref = batch[BatchKeys.INPUT_HEAD]
            input_tail_ref = batch[BatchKeys.INPUT_TAIL]
            input_rel_ref = batch[BatchKeys.INPUT_REL]

            input_head_tok, input_tail_tok, input_rel_tok = (
                self.tokenize(input_head_ref),
                self.tokenize(input_tail_ref),
                self.tokenize(input_rel_ref),
            )
            head_scores = self.get_scores(
                input_tokens=input_head_tok,
                schema_embeddings=schema_node_embeddings,
                chunk_size=node_loader.batch_size,
            )
            self.node_metrics.update(head_scores, gt[BatchKeys.HEAD])
            tail_scores = self.get_scores(
                input_tokens=input_tail_tok,
                schema_embeddings=schema_node_embeddings,
                chunk_size=node_loader.batch_size,
            )
            self.node_metrics.update(tail_scores, gt[BatchKeys.TAIL])
            rel_scores = self.get_scores(
                input_tokens=input_rel_tok,
                schema_embeddings=schema_edge_embeddings,
                chunk_size=edge_loader.batch_size,
            )
            self.edge_metrics.update(rel_scores, gt[BatchKeys.REL])

        node_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in self.node_metrics.compute().items()
        }
        edge_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in self.edge_metrics.compute().items()
        }
        self.print_stats(
            epoch=epoch,
            cumulated_loss=None,
            phase=ParameterKeys.VAL,
            node_metrics=node_metrics,
            edge_metrics=edge_metrics,
        )
        self.trigger = self.early_stop_callback(
            cumulated_loss=self.metric_mult
            * (node_metrics[self.metric_to_watch] + edge_metrics[self.metric_to_watch])
            / 2
        )

    @torch.no_grad()
    def test(self):
        if not self.test_loader:
            return {}
        self.model.load_state_dict(torch.load(self.early_stop.path))
        self.model = self.model.to(self.device)
        bar = self.get_bar(loader=self.test_loader, desc=f"{ParameterKeys.TEST}")
        self.rebuild_metrics(
            num_node_classes=self.test_loader.dataset.node_num_classes,
            num_rel_classes=self.test_loader.dataset.rel_num_classes,
        )
        self.model.eval()
        node_loader, edge_loader = self.test_loader.dataset.get_schema_items(
            batch_size=self.test_loader.batch_size,
            num_workers=self.test_loader.num_workers,
        )
        schema_node_embeddings, schema_edge_embeddings = self.get_schema_embeddings(
            node_loader=node_loader, edge_loader=edge_loader
        )
        node_predictions = set()
        rel_predictions = set()

        for ix, batch in bar:
            gt = batch.pop(BatchKeys.GT)
            head_qids = batch.pop(BatchKeys.HEAD_QID)
            rel_pids = batch.pop(BatchKeys.PID)
            tail_qids = batch.pop(BatchKeys.TAIL_QID)
            input_head_ref = batch[BatchKeys.INPUT_HEAD]
            input_tail_ref = batch[BatchKeys.INPUT_TAIL]
            input_rel_ref = batch[BatchKeys.INPUT_REL]

            input_head_tok, input_tail_tok, input_rel_tok = (
                self.tokenize(input_head_ref),
                self.tokenize(input_tail_ref),
                self.tokenize(input_rel_ref),
            )
            head_scores = self.get_scores(
                input_tokens=input_head_tok,
                schema_embeddings=schema_node_embeddings,
                chunk_size=node_loader.batch_size,
            )
            self.node_metrics.update(head_scores, gt[BatchKeys.HEAD])
            tail_scores = self.get_scores(
                input_tokens=input_tail_tok,
                schema_embeddings=schema_node_embeddings,
                chunk_size=node_loader.batch_size,
            )
            self.node_metrics.update(tail_scores, gt[BatchKeys.TAIL])
            rel_scores = self.get_scores(
                input_tokens=input_rel_tok,
                schema_embeddings=schema_edge_embeddings,
                chunk_size=edge_loader.batch_size,
            )
            self.edge_metrics.update(rel_scores, gt[BatchKeys.REL])

            head_predictions = set(
                [
                    (qid, self.test_loader.dataset.node_idx2cls_kb[x])
                    for (qid, x) in zip(head_qids, head_scores.argmax(dim=1).tolist())
                ]
            )
            tail_predictions = set(
                [
                    (qid, self.test_loader.dataset.node_idx2cls_kb[x])
                    for (qid, x) in zip(tail_qids, tail_scores.argmax(dim=1).tolist())
                ]
            )
            node_predictions = node_predictions | head_predictions | tail_predictions

            rel_batch_predictions = set(
                [
                    (pid, self.test_loader.dataset.rel_idx2cls_kb[x])
                    for (pid, x) in zip(
                        rel_pids.tolist(), rel_scores.argmax(dim=1).tolist()
                    )
                ]
            )
            rel_predictions = rel_predictions | rel_batch_predictions

        node_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in self.node_metrics.compute().items()
        }
        edge_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in self.edge_metrics.compute().items()
        }
        return dict(node=node_metrics, rel=edge_metrics), dict(
            node=list(node_predictions), edge=list(rel_predictions)
        )
