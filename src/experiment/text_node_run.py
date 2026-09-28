from .training_run import TrainingRun
from src.utils import ParameterKeys
from transformers import AutoTokenizer
from src.data.utils import BatchKeys
from src.models.utils import ReturnKeys
from torch.utils.data import DataLoader
import torch


class TextNodeRun(TrainingRun):
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

    def model_warmup(self, epoch: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.encoder.parameters():
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
        for batch in bar:
            input_node = batch[BatchKeys.INPUT_NODE]
            schema_ref = batch[BatchKeys.SCHEMA_NODE]
            input_tok = self.tokenize(input_node)
            schema_tok = self.tokenize(schema_ref)
            output_dict = self.model(input_tok, schema_tok)
            input_emb, target_emb = (
                output_dict[ReturnKeys.INPUT],
                output_dict[ReturnKeys.TARGET],
            )
            output_logits = input_emb @ target_emb.T
            labels = torch.arange(output_logits.size(0)).long().to(output_logits.device)
            loss = self.criterion(output_logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.detach().cpu().item()
            self.update_bar(bar=bar, loss=batch_loss)
            self.schedule(phase=ParameterKeys.TRAIN)

        cumulated_loss /= len(self.train_loader)
        self.print_stats(
            epoch=epoch, cumulated_loss=cumulated_loss, phase=ParameterKeys.TRAIN
        )

    def get_schema_embeddings(self, schema_nodes: DataLoader) -> torch.Tensor:
        # TODO:
        pass

    def get_scores(self, schema_loader: DataLoader) -> torch.Tensor:
        # TODO:
        pass

    @torch.no_grad()
    def val_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        self.metrics.reset()
        self.model.eval()

        schema_nodes = self.val_loader.dataset.get_schema_nodes(
            batch_size=self.val_loader.batch_size,
            num_workers=self.val_loader.num_workers,
        )
        schema_node_embeddings = self.get_schema_embeddings(schema_nodes)

        for batch in bar:
            gt = batch.pop(BatchKeys.GT)
            input_text = batch[BatchKeys.INPUT_NODE]
            input_tok = self.tokenize(input_text)
            scores = self.get_scores(
                input_tok, schema_node_embeddings, chunk_size=schema_nodes.batch_size
            )
            self.metrics.update(scores, gt)

        metrics = self.metrics.compute()
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
        return super().test()
