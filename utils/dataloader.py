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
    
def get_image_transforms(image_size: Tuple[int, int], mode: str):
    """
    Modes of augmentation:
    1. 'none': No augmentation, just resizing and normalization.
    2. 'standard': General geometric and color augmentations.
    3. 'enhanced': CLAHE + Sharpening + General augmentations.
    4. 'val': Strict resizing and normalization for validation/testing.
    """

    base_norm = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]

    if mode == 'none' or mode == 'val':
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            *base_norm
        ])

    elif mode == 'standard':
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            *base_norm
        ])
        
    elif mode == 'enhanced':
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            # CLAHE: clip_limit 2.0, tile_grid 8x8
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            # Unsharp Masking via Sharpen: alpha 1.0 matches weight amount
            A.Sharpen(alpha=(1.0, 1.0), lightness=(1.0, 1.0), p=1.0),
            # Followed by standard training augmentations
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            *base_norm
        ])
        
    # Default fallback
    return A.Compose([A.Resize(image_size[0], image_size[1]), *base_norm])

def save_augmentation_samples(train_loader, save_dir, num_samples=10):
    if num_samples <= 0:
        return
    
    # Define pipelines WITHOUT Normalization for visualization
    pipes = {
        "none": A.Compose([A.Resize(224, 224)]),
        "standard": A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=15, p=1.0), # Force rotation for sample view
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=0.5)
        ]),
        "enhanced": A.Compose([
            A.Resize(224, 224),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Sharpen(alpha=(1.0, 1.0), lightness=(1.0, 1.0), p=1.0),
            A.Rotate(limit=15, p=1.0)
        ])
    }

    # Create subdirectories for each type
    for mode_name in pipes.keys():
        path = os.path.join(save_dir, 'aug_samples', mode_name)
        os.makedirs(path, exist_ok=True)

    print(f"INFO: Saving {num_samples} samples per augmentation type to {save_dir}/aug_samples/")

    dataset = train_loader.dataset
    # We use a small loop to save individual images
    for i in range(min(num_samples, len(dataset))):
        img_path, label_idx = dataset.samples[i]
        class_name = dataset.class_names[label_idx]
        
        # Load the raw image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for mode_name, pipe in pipes.items():
            transformed = pipe(image=image)['image']
            
            # Save the image (convert back to BGR for OpenCV save)
            save_img = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
            filename = f"sample_{i}_{class_name}.jpg"
            save_path = os.path.join(save_dir, 'aug_samples', mode_name, filename)
            cv2.imwrite(save_path, save_img)
            
def create_data_loaders(
    data_dir: str, 
    batch_size: int = 32, 
    aug_type: str = 'none', 
    class_names: Optional[List[str]] = None, 
    image_size: Tuple[int, int] = (224, 224)
):
    # Scan disk ONLY ONCE
    full_ds = RoadDamageDataset(data_dir, subset='full', class_names=class_names, image_size=image_size)
    
    # 80% Train, 20% Temp (Val/Test)
    train_idx, temp_idx = train_test_split(
        range(len(full_ds)), 
        train_size=0.8, 
        stratify=full_ds.targets, 
        random_state=42
    )
    
    # Split Temp into 50% Val, 50% Test (10% of total each)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=0.5, 
        stratify=[full_ds.targets[i] for i in temp_idx],
        random_state=42
    )

    def build_subset(indices, subset_name):
        # Use the requested aug_type for training, otherwise use 'val' for Val/Test
        mode = aug_type if subset_name == 'train' else 'val'
        
        ds = RoadDamageDataset(
            data_dir, 
            transform=get_image_transforms(image_size, mode), 
            subset=subset_name, 
            class_names=full_ds.class_names
        )
        ds.samples = [full_ds.samples[i] for i in indices]
        ds.targets = [s[1] for s in ds.samples]
        ds.log_summary()
        
        return DataLoader(ds, batch_size=batch_size, shuffle=(subset_name == 'train'))

    train_loader = build_subset(train_idx, 'train')
    val_loader = build_subset(val_idx, 'val')
    test_loader = build_subset(test_idx, 'test')

    return train_loader, val_loader, test_loader, full_ds.class_weights