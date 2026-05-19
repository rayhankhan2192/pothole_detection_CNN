import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoadDamageDataset(Dataset):
    """
    Dataset loader for Crack, Pothole, and Surface erosion images.
    """
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        subset: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        class_names: Optional[List[str]] = None
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset
        self.image_size = image_size

        # Set class names once based on folder structure
        if class_names is not None:
            self.class_names = class_names  
        else:
            self.class_names = sorted([
                entry for entry in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, entry))
            ])

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Only perform heavy disk operations if we are the 'full' dataset
        if subset == 'full':
            self.samples = self.load_samples()
            self.targets = [s[1] for s in self.samples]
            self.class_weights = self.calculate_class_weights()
        else:
            self.samples = []
            self.targets = []
            self.class_weights = None