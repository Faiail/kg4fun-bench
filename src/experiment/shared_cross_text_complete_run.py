from .shared_text_complete_run import SharedTextCompleteRun
from .utils import ParameterKeys
from src.data.utils import BatchKeys
from src.models.utils import ReturnKeys
from torch.nn import functional as F
import torch


class SharedCrossTextCompleteRun(SharedTextCompleteRun):
    def model_warmup(self, epoch):
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.encoder.parameters():
            p.requires_grad = epoch > self.num_warmup_epochs

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
            scores = self.model(
                input_head=input_head_tok,
                input_tail=input_tail_tok,
                input_rel=input_rel_tok,
                target_head=target_head_tok,
                target_tail=target_tail_tok,
                target_rel=target_rel_tok,
            )
            labels = batch.get(
                BatchKeys.GT,
                {k: torch.arange(v.size(0)).long() for k, v in scores.items()},
            )
            labels = {k: v.to(self.device) for k, v in labels.items()}
            loss = self.criterion(scores, labels)

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
    def get_spec_schema_embeddings(self, loader, desc):
        all_cls = []
        all_tokens = []
        all_masks = []
        bar = self.get_bar(loader=loader, desc=desc)
        batch_size = loader.batch_size
        for ix, batch in bar:
            schema_ref = batch[BatchKeys.SCHEMA]
            schema_tokens = self.tokenize(schema_ref)
            schema_embedds = self.model.encode(schema_tokens)
            all_cls.append(schema_embedds[ReturnKeys.CLS].cpu())
            all_tokens.append(schema_embedds[ReturnKeys.TOKEN].cpu())
            all_masks.append(schema_embedds[ReturnKeys.MASK].cpu())

        # Dynamically pad to the max token length across all schema items
        max_len = max(t.size(1) for t in all_tokens)
        padded_tokens = []
        padded_masks = []
        for t, m in zip(all_tokens, all_masks):
            pad_len = max_len - t.size(1)
            if pad_len > 0:
                t = F.pad(t, (0, 0, 0, pad_len), value=0.0)
                m = F.pad(m, (0, pad_len), value=0.0)
            padded_tokens.append(t)
            padded_masks.append(m)

        return {
            ReturnKeys.CLS: torch.cat(all_cls, dim=0),
            ReturnKeys.TOKEN: torch.cat(padded_tokens, dim=0),
            ReturnKeys.MASK: torch.cat(padded_masks, dim=0),
        }        

    @torch.no_grad()
    def get_schema_embeddings(self, node_loader, edge_loader):
        return self.get_spec_schema_embeddings(
            loader=node_loader, desc="Get node classes item embeddings", task="node"
        ), self.get_spec_schema_embeddings(
            loader=edge_loader, desc="Get edge rels item embeddings", task="edge"
        )

    @torch.no_grad()
    def get_scores(
        self,
        input_tokens: dict,
        task: str,
        schema_embeddings: dict[str, torch.Tensor],
        chunk_size: int = 1,
    ) -> torch.Tensor:
        len_schema_nodes = schema_embeddings[ReturnKeys.CLS].size(0)
        input_embedds = self.model.encode(input_tokens)
        scores = list()
        for start_pos in range(0, len_schema_nodes, chunk_size):
            end_pos = min(start_pos + chunk_size, len_schema_nodes)
            chunk_schema_embeddings = {
                k: v[start_pos:end_pos].to(self.device)
                for k, v in schema_embeddings.items()
            }
            chunk_scores = self.model.match(input_embedds, chunk_schema_embeddings, unit=task)
            scores.append(chunk_scores.cpu())
        scores = torch.cat(scores, dim=-1)
        return scores

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
                task=ReturnKeys.NODE,
            )
            self.node_metrics.update(head_scores, gt[BatchKeys.HEAD])
            tail_scores = self.get_scores(
                input_tokens=input_tail_tok,
                schema_embeddings=schema_node_embeddings,
                chunk_size=node_loader.batch_size,
                task=ReturnKeys.NODE,
            )
            self.node_metrics.update(tail_scores, gt[BatchKeys.TAIL])
            rel_scores = self.get_scores(
                input_tokens=input_rel_tok,
                schema_embeddings=schema_edge_embeddings,
                chunk_size=edge_loader.batch_size,
                task=ReturnKeys.EDGE,
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
        n_val = max(node_metrics[self.metric_to_watch], 1e-6)
        e_val = max(edge_metrics[self.metric_to_watch], 1e-6)
        balanced_score = 2.0 * (n_val * e_val) / (n_val + e_val)
        self.trigger = self.early_stop_callback(
            cumulated_loss=self.metric_mult * balanced_score
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
                task=ReturnKeys.NODE,
            )
            self.node_metrics.update(head_scores, gt[BatchKeys.HEAD])
            tail_scores = self.get_scores(
                input_tokens=input_tail_tok,
                schema_embeddings=schema_node_embeddings,
                chunk_size=node_loader.batch_size,
                task=ReturnKeys.NODE,
            )
            self.node_metrics.update(tail_scores, gt[BatchKeys.TAIL])
            rel_scores = self.get_scores(
                input_tokens=input_rel_tok,
                schema_embeddings=schema_edge_embeddings,
                chunk_size=edge_loader.batch_size,
                task=ReturnKeys.EDGE,
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