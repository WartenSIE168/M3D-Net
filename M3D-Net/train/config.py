import torch
import torchvision

# Parameter configuration
class_names = ['RiskBehavior', 'Distracted', 'Normallevel', 'HighlyCentralized', 'Fatigue', 'Yawning']
num_classes = len(class_names)
sequence_length = 90  # Temporal sequence length

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset split ratio
split_ratios = [0.7, 0.2, 0.1]  # Training set / Validation set / Test set
random_seed = 42  # Random seed

# Data augmentation configuration
data_transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.6, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}    