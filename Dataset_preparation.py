import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from Model import Swin_T  # Assuming your Swin Transformer is imported from a Model.py file


# Custom Image Dataset (Using already loaded data from the folder)
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_map, transform=None):
        self.data = []
        self.label_map = label_map
        self.transform = transform

        # Read images and labels
        for class_folder in os.listdir(img_dir):
            class_folder_path = os.path.join(img_dir, class_folder)
            if os.path.isdir(class_folder_path):
                for img_path in glob.glob(os.path.join(class_folder_path, '*.jpg')):
                    img = Image.open(img_path)
                    self.data.append((img, class_folder))  # Store image and folder name as label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        label = self.label_map[label]  # Convert folder name to numeric label
        return image, label


# Function to Train the Model
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """Training loop for the model."""
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Track loss and accuracy
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

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


# Plotting the learning curves
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


# Data Preparation and Transformation
train_data_path = "train/"
val_data_path = "val/"
img_size = (224, 224)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare label mapping (Folder names as class labels)
label_map = {class_folder: idx for idx, class_folder in enumerate(os.listdir(train_data_path))}

# Create Dataset and DataLoader for both training and validation
image_datasets = {
    'train': CustomImageDataset(train_data_path, label_map, transform),
    'val': CustomImageDataset(val_data_path, label_map, transform)
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=1),
    'val': DataLoader(image_datasets['val'], batch_size=4, shuffle=False, num_workers=1)
}

# Initialize the Model, Loss, Optimizer, and Scheduler
model = Swin_T(num_classes=len(label_map))  # Set num_classes to the number of classes in the label_map

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Train the Model
model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=device)
