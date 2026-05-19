import torch
import torch.nn as nn
from torchvision import models

class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(CustomMobileNetV2, self).__init__()
        
        # Load pre-trained MobileNetV2 (weights='imagenet')
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Freeze the base model parameters (base.trainable = False)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # MobileNetV2's feature extractor outputs 1280 features
        in_features = self.base_model.last_channel 
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)