import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(EfficientNetB0, self).__init__()
        
        # Load pre-trained EfficientNetB0 (weights='imagenet')
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze the base model parameters (base.trainable = False)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # EfficientNetB0's feature extractor outputs 1280 features
        in_features = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # The base_model automatically applies GlobalAveragePooling2D 
        # before passing data to the classifier we just built.
        return self.base_model(x)