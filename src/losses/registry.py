from .info_nce_loss import InfoNCELoss
from .adaptive_arcface_loss import AdaptiveArcFaceLoss
from .cosface_loss import CosFaceLoss
import torch


LOSS_REGISTRY = {
    "InfoNCELoss": InfoNCELoss,
    "AdaptiveArcFaceLoss": AdaptiveArcFaceLoss,
    "CosFaceLoss": CosFaceLoss,
    **torch.nn.__dict__,
}