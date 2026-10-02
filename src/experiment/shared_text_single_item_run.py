from .training_run import TrainingRun
from src.utils import ParameterKeys
from transformers import AutoTokenizer
from src.data.utils import BatchKeys
from src.models.utils import ReturnKeys
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from copy import deepcopy
import torch


class SharedTextSingleItemRun(TrainingRun):
    def init(self):
        super().init()
        print("Init Tokenizer...")
        self._init_tokenizer()
        print("Done!")

    def _init_tokenizer(self) -> None:
        tokenizer_parameters = self.parameters.get(ParameterKeys.TOKENIZER)
        tokenizer_cfg = tokenizer_parameters.get(ParameterKeys.CFG, dict())
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg)
        self.tokenizer_kwargs = tokenizer_parameters.get(ParameterKeys.KWARGS, dict())

    def rebuild_metrics(self, num_classes: int) -> MetricCollection:
        metric_parameters = deepcopy(self.parameters.get(ParameterKeys.METRICS))
        for metric in metric_parameters:
            if (
                "num_classes"
                in metric_parameters[metric].get(ParameterKeys.CFG, dict()).keys()
            ):
                metric_parameters[metric][ParameterKeys.CFG][
                    "num_classes"
                ] = num_classes
        self.parameters[ParameterKeys.METRICS] = metric_parameters
        self._init_metrics()

    def model_warmup(self, epoch: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.encoder.encoder.layer[-2:].parameters():
            p.requires_grad = epoch > self.num_warmup_epochs
        for p in self.model.encoder.pooler.parameters():
            p.requires_grad = epoch > self.num_warmup_epochs

    def tokenize(self, text: str | list[str]) -> dict:
        return self.tokenizer(text, **self.tokenizer_kwargs).to(self.device)

    def train_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAIN} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.train()
        self.model_warmup(epoch=epoch)
        for ix, batch in bar:
            input_ref = batch[BatchKeys.INPUT]
            schema_ref = batch[BatchKeys.SCHEMA]
            input_tok = self.tokenize(input_ref)
            schema_tok = self.tokenize(schema_ref)
            output_dict = self.model(input_tok, schema_tok)
            input_emb, target_emb = (
                output_dict[ReturnKeys.INPUT],
                output_dict[ReturnKeys.TARGET],
            )
            output_logits = input_emb @ target_emb.T
            labels = torch.arange(output_logits.size(0)).long().to(output_logits.device)
            loss = self.criterion(output_logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar=bar, loss=batch_loss)
            self.schedule(phase=ParameterKeys.TRAIN)

        cumulated_loss /= len(self.train_loader)
        self.print_stats(
            epoch=epoch, cumulated_loss=cumulated_loss, phase=ParameterKeys.TRAIN
        )

    @torch.no_grad()
    def get_schema_embeddings(self, schema_loader: DataLoader) -> torch.Tensor:
        schema_item_embeddings = torch.empty(
            size=(len(schema_loader.dataset), self.model.schema_hidden_size)
        )
        bar = self.get_bar(loader=schema_loader, desc="Get schema item embeddings")
        batch_size = schema_loader.batch_size
        for ix, batch in bar:
            schema_ref = batch[BatchKeys.SCHEMA]
            schema_tokens = self.tokenize(schema_ref)
            schema_embedds = self.model.encode(schema_tokens)
            schema_item_embeddings[(ix * batch_size) : ((ix + 1) * batch_size)] = (
                schema_embedds.cpu()
            )
        return schema_item_embeddings

    @torch.no_grad()
    def get_scores(
        self,
        input_tokens: dict,
        schema_embeddings: torch.Tensor,
        chunk_size: int = 1,
    ) -> torch.Tensor:
        len_schema_nodes = schema_embeddings.size(0)
        input_embedds = self.model.encode(input_tokens)
        scores = list()
        for idx in range(0, (len_schema_nodes // chunk_size) + 1):
            start_pos = idx * chunk_size
            end_pos = (idx + 1) * chunk_size

            chunk_schema_embeddings = schema_embeddings[start_pos:end_pos].to(
                self.device
            )
            chunk_scores = input_embedds @ chunk_schema_embeddings.T
            scores.append(chunk_scores.cpu())
        scores = torch.cat(scores, dim=-1)
        return scores

    @torch.no_grad()
    def val_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        self.rebuild_metrics(self.val_loader.dataset.num_classes)
        self.model.eval()

        schema_items = self.val_loader.dataset.get_schema_items(
            batch_size=self.val_loader.batch_size,
            num_workers=self.val_loader.num_workers,
        )
        schema_item_embeddings = self.get_schema_embeddings(schema_items)

        for ix, batch in bar:
            gt = batch.pop(BatchKeys.GT)
            input_text = batch[BatchKeys.INPUT]
            input_tok = self.tokenize(input_text)
            scores = self.get_scores(
                input_tok, schema_item_embeddings, chunk_size=schema_items.batch_size
            )
            self.metrics.update(scores, gt)

        metrics = self.metrics.compute()
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }
        self.print_stats(
            epoch=epoch,
            cumulated_loss=None,
            phase=ParameterKeys.VAL,
            metrics=metrics,
        )

        self.trigger = self.early_stop_callback(
            cumulated_loss=self.metric_mult * metrics[self.metric_to_watch]
        )

    def test(self):
        if not self.test_loader:
            return {}
        self.model.load_state_dict(torch.load(self.early_stop.path))
        self.model = self.model.to(self.device)
        bar = self.get_bar(
            loader=self.test_loader,
            desc=f"{ParameterKeys.TEST}",
        )
        self.rebuild_metrics(self.test_loader.dataset.num_classes)
        self.model.eval()
        schema_items = self.test_loader.dataset.get_schema_items(
            batch_size=self.test_loader.batch_size,
            num_workers=self.test_loader.num_workers,
        )
        schema_item_embeddings = self.get_schema_embeddings(schema_items)
        predictions = list()
        for ix, batch in bar:
            gt = batch[BatchKeys.GT]
            qids = batch[BatchKeys.QID]
            input_text = batch[BatchKeys.INPUT]
            input_tok = self.tokenize(input_text)
            scores = self.get_scores(
                input_tok, schema_item_embeddings, chunk_size=schema_items.batch_size
            )
            batch_predictions = [
                (qid, self.test_loader.dataset.idx2cls_kb[x])
                for (qid, x) in zip(qids, scores.argmax(dim=1).tolist())
            ]
            predictions.extend(batch_predictions)
            self.metrics.update(scores, gt)

        metrics = self.metrics.compute()
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }
        return metrics, predictions
