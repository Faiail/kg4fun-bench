from .info_nce_loss import InfoNCELoss
import torch



LOSS_REGISTRY = {"InfoNCELoss": InfoNCELoss, **torch.nn.__dict__}