from .shared_text_single_item_run import SharedTextSingleItemRun
from src.data.utils import BatchKeys
from torch.utils.data import DataLoader
import torch


class DedicatedTextSingleItemRun(SharedTextSingleItemRun):
    def model_warmup(self, epoch: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.input_encoder.parameters():
            p.requires_grad = False
        for p in self.model.target_encoder.parameters():
            p.requires_grad = False
        if hasattr(self.model.input_encoder, "encoder"):
            for p in self.model.input_encoder.encoder.layer[-2:].parameters():
                p.requires_grad = epoch > self.num_warmup_epochs
        else:
            for p in self.model.input_encoder.layers[-2:].parameters():
                p.requires_grad = epoch > self.num_warmup_epochs
        if hasattr(self.model.target_encoder, "encoder"):
            for p in self.model.target_encoder.encoder.layer[-2:].parameters():
                p.requires_grad = epoch > self.num_warmup_epochs
        else:
            for p in self.model.target_encoder.layers[-2:].parameters():
                p.requires_grad = epoch > self.num_warmup_epochs

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
            schema_embedds = self.model.encode_target(schema_tokens)
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
        input_embedds = self.model.encode_input(input_tokens)
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