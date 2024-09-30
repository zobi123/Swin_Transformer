# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:40:07 2024

@author: MSP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
from PIL import Image
import matplotlib.pyplot as plt
from Model import Swin_T  # Assuming your Swin Transformer model is in a file called Model.py


# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = annotations_file  # Assuming annotations_file is a DataFrame or similar structure
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # Assuming first column is image paths
        image = Image.open(img_path).convert("RGB")  # Open the image
        label = self.img_labels.iloc[idx, 1]  # Assuming second column is the label
        
        if self.transform:
            image = self.transform(image)

        return image, label  # Ensure only two values are returned (image, label)


# Training Loop
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """Training loop for the model."""
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Track loss and accuracy
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []  # Fixed initialization

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store losses and accuracy for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())  # .item() to convert tensor to float
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())  # .item() to convert tensor to float

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot the learning curves
    plot_learning_curve(train_losses, val_losses, train_accs, val_accs)

    return model


# Function to Plot Learning Curve
def plot_learning_curve(train_losses, val_losses, train_accs, val_accs):
    """Function to plot training and validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# Data Preparation (Transformations and DataLoader)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Assuming that 'annotations_file' is a DataFrame or similar that contains image paths and labels.
data_dir = '\dataset"'
annotations_file = 'your_annotations_dataframe'  # Placeholder

# Create Dataset and DataLoader for both training and validation
image_datasets = {x: CustomImageDataset(annotations_file, os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Initialize the Model, Loss, Optimizer, and Scheduler
model = Swin_T(num_classes=4)  # Assuming you have 4 classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Train the Model
model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=device)
