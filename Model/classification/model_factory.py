import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from Model.classification.customcnn import CustomCNN
from Model.classification.efficientnet import EfficientNetB0
from Model.classification.mobileNetV2 import CustomMobileNetV2
from Model.classification.resNet50 import ResNet50
from Model.classification.vgg19 import VGG19

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model(model_name: str, num_classes: int = 3, **kwargs):
    """
    Factory function to return the specified model.
    """
    model_name = model_name.lower()
    
    if model_name == 'customcnn':
        logger.info(f"Initializing CustomCNN with {num_classes} classes.")
        return CustomCNN(num_classes=num_classes)
        
    elif model_name == 'efficientnet':
        logger.info(f"Initializing Frozen EfficientNetB0 with {num_classes} classes.")
        return EfficientNetB0(num_classes=num_classes)
        
    elif model_name == 'mobilenetv2':
        logger.info(f"Initializing CustomMobileNetV2 with {num_classes} classes.")
        return CustomMobileNetV2(num_classes=num_classes)
    elif model_name == 'resnet50':
        logger.info(f"Initializing Frozen ResNet50 with {num_classes} classes.")
        return ResNet50(num_classes=num_classes)
    elif model_name == 'vgg19':
        logger.info(f"Initializing Frozen VGG19 with {num_classes} classes.")
        return VGG19(num_classes=num_classes)
        
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Use 'customcnn', 'efficientnet', or 'mobilenetv2'.")
    
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for image classification
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss