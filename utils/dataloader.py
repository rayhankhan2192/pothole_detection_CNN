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
            
    def load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.class_names:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path): continue
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((os.path.join(class_path, img_name), self.class_to_idx[class_name]))
        return samples

    def calculate_class_weights(self) -> torch.Tensor:
        class_counts = np.zeros(len(self.class_names), dtype=np.int64)
        for t in self.targets:
            class_counts[t] += 1
        total = len(self.targets)
        weights = [total / (len(self.class_names) * c) if c > 0 else 0.0 for c in class_counts]
        return torch.tensor(weights, dtype=torch.float32)

    def log_summary(self):
        logger.info(f"Dataset Summary: {self.subset}")
        logger.info(f"Total samples: {len(self.samples)}")
        counts = np.bincount(self.targets, minlength=len(self.class_names))
        for name, count in zip(self.class_names, counts):
            logger.info(f"  {name}: {count}")

    def __len__(self) -> int: 
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize handled by albumentations if transform exists, but keeping fallback
        if not self.transform:
            image = cv2.resize(image, self.image_size)
            
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label