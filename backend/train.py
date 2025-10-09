import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
import random
import os

# Configuration
class Config:
    # Data
    data_dir = "../datasets/plant_disease_recognition/train"
    val_dir = "../datasets/plant_disease_recognition/validation"  # Add validation directory
    num_workers = 4
    
    # Training
    batch_size = 32
    num_epochs = 50  # More epochs for better accuracy
    learning_rate = 0.0003  # Lower learning rate for better convergence
    weight_decay = 1e-4
    warmup_epochs = 3  # Learning rate warmup
    early_stop_patience = 8  # More patience for better training
    
    # Model
    model_name = 'efficientnet-b1'  # Better accuracy than b0
    num_classes = 39  # Will be updated based on dataset
    dropout = 0.3  # Higher dropout for better regularization
    
    # Paths
    model_dir = "models"
    model_path = os.path.join(model_dir, "best_model.pth")
    class_to_idx_path = os.path.join(model_dir, "class_to_idx.json")
    
    # Augmentation
    image_size = 240  # Slightly larger for better feature extraction
    
    def __init__(self):
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get class names and create mapping
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data augmentation and transforms
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(Config.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(Config.image_size + 64),
            transforms.CenterCrop(Config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

# Model
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        # Load pre-trained EfficientNet from torchvision
        self.model = models.efficientnet_b1(weights='DEFAULT')
        
        # Replace the classifier head with dropout
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=Config.dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Initialize weights for the new head
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0
    epochs_no_improve = 0
    
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler() if 'cuda' in str(device) else None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Learning rate warmup
        if epoch < Config.warmup_epochs:
            lr_scale = (epoch + 1) / Config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = Config.learning_rate * lr_scale
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar
            pbar = tqdm(dataloaders[phase], desc=f'Epoch {epoch+1} {phase}')
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        if scaler:
                            scaler.scale(loss).backward()
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{running_loss / ((pbar.n + 1) * inputs.size(0)):.4f}",
                    'acc': f"{100 * running_corrects.double() / ((pbar.n + 1) * inputs.size(0)):.2f}%"
                })
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save best model
            if phase == 'val':
                if epoch_acc > best_acc:
                    print(f'Validation accuracy improved from {best_acc:.4f} to {epoch_acc:.4f}. Saving model...')
                    best_acc = epoch_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'accuracy': epoch_acc,
                    }, Config.model_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'No improvement in validation accuracy for {epochs_no_improve} epochs')
                    
                    # Early stopping
                    if epochs_no_improve >= Config.early_stop_patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        return model
        
        # Step the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()
    
    return model

def main():
    # Initialize config
    config = Config()
    device = config.device
    print(f'Using device: {device}')
    
    # Data loading - separate train and validation datasets
    train_dataset = PlantDiseaseDataset(
        config.data_dir,
        transform=get_transforms(train=True)
    )
    
    # Use separate validation dataset if available
    if os.path.exists(config.val_dir):
        val_dataset = PlantDiseaseDataset(
            config.val_dir,
            transform=get_transforms(train=False)
        )
        print(f'Using separate validation dataset with {len(val_dataset)} samples')
    else:
        # Fallback to splitting the training data
        print('No separate validation directory found, splitting training data...')
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Update num_classes based on dataset
    config.num_classes = len(train_dataset.classes)
    print(f'Found {config.num_classes} classes: {train_dataset.classes}')
    
    # Create data loaders with weighted sampling for class imbalance
    train_weights = [1.0 / len(train_dataset) for _ in range(len(train_dataset))]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_weights, num_samples=len(train_weights), replacement=True
    )
    
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    }
    
    # Initialize model
    model = PlantDiseaseModel(config.num_classes).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model initialized with {total_params:,} trainable parameters')
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(dataloaders['train']),
        epochs=config.num_epochs,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Train the model
    print('Starting training...')
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.num_epochs,
        device=device
    )
    
    # Save class to index mapping and class names
    save_data = {
        'class_to_idx': train_dataset.class_to_idx,
        'classes': train_dataset.classes,
        'model_name': config.model_name,
        'image_size': config.image_size,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }
    
    with open(config.class_to_idx_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print('\nTraining complete!')
    print(f'Best model saved to: {os.path.abspath(config.model_path)}')
    print(f'Class mapping saved to: {os.path.abspath(config.class_to_idx_path)}')

if __name__ == "__main__":
    main()
