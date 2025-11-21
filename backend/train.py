import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import json
from pathlib import Path
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration
class Config:
    # Data
    
    data_dir = "../datasets/plant_disease_recognition/train"
    val_dir = "../datasets/plant_disease_recognition/validation"
    test_dir = "../datasets/plant_disease_recognition/test"
    
    # System
    num_workers = 4
    pin_memory = True
    
    # Training
    batch_size = 64  # Reduced for RTX 3050 (4GB/6GB VRAM) with larger model
    num_epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4
    label_smoothing = 0.1
    
    # Model
    model_name = 'efficientnet_b2' # Good balance of speed/accuracy for RTX 3050
    dropout = 0.4
    
    # Paths
    model_dir = "models"
    model_path = os.path.join(model_dir, "best_model.pth")
    class_to_idx_path = os.path.join(model_dir, "class_to_idx.json")
    
    # Augmentation
    image_size = 260 # EfficientNet-B2 native resolution
    
    def __init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get class names
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random other image on error
            return self.__getitem__(random.randint(0, len(self) - 1))

# Advanced Augmentation
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((Config.image_size + 32, Config.image_size + 32)),
            transforms.RandomResizedCrop(Config.image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Model Definition
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b2'):
        super(PlantDiseaseModel, self).__init__()
        
        # Load pretrained model
        if model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(weights='DEFAULT')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity() # Remove original classifier
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='DEFAULT')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        # Custom Head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(Config.dropout / 2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
            
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
    return running_loss / len(loader), 100. * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def main():
    config = Config()
    print(f"Using device: {config.device}")
    
    # Datasets
    print("Loading datasets...")
    train_dataset = PlantDiseaseDataset(config.data_dir, get_transforms(train=True))
    val_dataset = PlantDiseaseDataset(config.val_dir, get_transforms(train=False))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Save class mapping immediately
    save_data = {
        'class_to_idx': train_dataset.class_to_idx,
        'classes': train_dataset.classes,
        'model_name': config.model_name
    }
    with open(config.class_to_idx_path, 'w') as f:
        json.dump(save_data, f, indent=2)
        
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Model Setup
    model = PlantDiseaseModel(len(train_dataset.classes), config.model_name).to(config.device)
    
    # Training Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Training Loop
    best_acc = 0.0
    patience = 7
    counter = 0
    
    print("\nStarting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, config.device, scaler
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            print(f"New best model! ({best_acc:.2f}% -> {val_acc:.2f}%)")
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'class_to_idx': train_dataset.class_to_idx
            }, config.model_path)
            counter = 0
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")
            
        if counter >= patience:
            print("Early stopping triggered!")
            break
            
    print("Training complete!")

if __name__ == "__main__":
    main()
