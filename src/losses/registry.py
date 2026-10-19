from .info_nce_loss import InfoNCELoss
from .adaptive_arcface_loss import AdaptiveArcFaceLoss
import torch


LOSS_REGISTRY = {
    "InfoNCELoss": InfoNCELoss,
    "AdaptiveArcFaceLoss": AdaptiveArcFaceLoss,
    **torch.nn.__dict__,
}