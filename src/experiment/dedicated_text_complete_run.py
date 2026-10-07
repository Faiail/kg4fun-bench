from .shared_text_complete_run import SharedTextCompleteRun
from src.data.utils import BatchKeys
from torch.utils.data import DataLoader
import torch


class DedicatedTextCompleteRun(SharedTextCompleteRun):

    torch.no_grad()
    def get_spec_schema_embeddings(self, loader: DataLoader, desc: str):
        schema_item_embeddings = torch.empty(
            size=(len(loader.dataset), self.model.schema_hidden_size)
        )
        bar = self.get_bar(loader=loader, desc=desc)
        batch_size = loader.batch_size
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
