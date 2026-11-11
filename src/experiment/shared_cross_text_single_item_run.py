from .shared_text_single_item_run import SharedTextSingleItemRun
from src.models.utils import ReturnKeys
from src.data.utils import BatchKeys
from .utils import ParameterKeys
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


class SharedCrossTextSingleItemRun(SharedTextSingleItemRun):
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
            input_ref = batch[BatchKeys.INPUT]
            schema_ref = batch[BatchKeys.SCHEMA]

            input_tok = self.tokenize(input_ref)
            schema_tok = self.tokenize(schema_ref)
            scores = self.model(input_tok, schema_tok)
            labels = batch.get(BatchKeys.GT, torch.arange(scores.size(0)).long()).to(
                self.device
            )
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
    def get_schema_embeddings(self, schema_loader: DataLoader) -> dict[str, torch.Tensor]:
        all_cls = []
        all_tokens = []
        all_masks = []
        bar = self.get_bar(loader=schema_loader, desc="Get schema item embeddings")
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
    def get_scores(
        self,
        input_tokens: dict,
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
            chunk_scores = self.model.match(input_embedds, chunk_schema_embeddings)
            scores.append(chunk_scores.cpu())
        scores = torch.cat(scores, dim=-1)
        return scores
