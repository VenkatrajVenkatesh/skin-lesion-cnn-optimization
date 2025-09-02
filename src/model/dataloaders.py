import os
import cv2
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from augumentation import get_train_transforms, get_valid_transforms
 
def load_config(path="config.yaml"):
     # Load configuration parameters from a YAML file
    with open(path, 'r') as file:
        return yaml.safe_load(file)
 
class ImageClassificationDataset(Dataset):
    def __init__(self, df, transform=None, label_map=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map or {}
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['ImageFp']
        label_str = row['label']
 
        # Convert label string to integer using the label map
        label_id = self.label_map.get(label_str, -1)
        # One-hot encode the label
        one_hot = np.zeros(len(self.label_map), dtype=np.float32)
        if label_id != -1:
            one_hot[label_id] = 1.0
 
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
        # Use black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply augmentations/transforms
        if self.transform:
            image = self.transform(image=image)['image']
        # Return image tensor and one-hot encoded label
        return image, torch.tensor(one_hot)
 
def get_dataloaders(processed_df, config ):
        
        # Load parameters from config
        img_size = config["img_size"]
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
        label_map = config["label_map"]
        
        # Split data into training and validation sets
        train_df = processed_df[processed_df['Split'] == 0].copy()
        val_df = processed_df[processed_df['Split'] == 1].copy()

        # Create datasets with appropriate transforms
        train_dataset = ImageClassificationDataset(
            train_df,
            transform=get_train_transforms(img_size),
            label_map=label_map
            )
 
        val_dataset = ImageClassificationDataset(
            val_df,
            transform=get_valid_transforms(img_size),
           label_map=label_map
     )
          # Compute class weights for imbalanced dataset handling
        train_targets = [label_map[label] for label in train_df['label']]
        class_sample_count = np.array([train_targets.count(i) for i in range(len(label_map))])
        weights = 1. / class_sample_count
        samples_weights = np.array([weights[t] for t in train_targets])
        samples_weights = torch.tensor(samples_weights, dtype=torch.float)
 
        # Create weighted sampler for balanced training
        sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
        use_sampler = config.get("use_sampler",True)

         # Create DataLoader for training
        if use_sampler:
            print("using weighted dataloader")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers ,pin_memory = True)
        else:
            print(" not using weighted dataloader")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True, num_workers=num_workers,pin_memory = True)

        # Validation DataLoader (no shuffling)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
        return train_loader, val_loader