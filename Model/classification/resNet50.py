import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(ResNet50, self).__init__()
        
        # Load pre-trained ResNet50 (weights='imagenet')
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze the base model parameters (base.trainable = False)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # ResNet50's feature extractor outputs 2048 features
        in_features = self.base_model.fc.in_features
        
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)