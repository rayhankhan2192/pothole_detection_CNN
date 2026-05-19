import torch
import torch.nn as nn
from torchvision import models

class CustomVGG19(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(CustomVGG19, self).__init__()
        
        # Load pre-trained VGG19 (weights='imagenet')
        self.base_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Freeze the base model parameters (base.trainable = False)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 1. Replicate Keras `GlobalAveragePooling2D`
        # PyTorch defaults to 7x7 pooling for VGG, so we override it to 1x1
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.base_model.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.base_model(x)