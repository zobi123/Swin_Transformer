from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from Model import Swin_T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Function to compute confusion matrix, classification metrics, and save predicted vs actual images
def compute_metrics_and_save_images(model, dataloader, device, class_names, save_dir='predictions'):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    class_image_saved = {class_name: False for class_name in class_names}  # Track if an image is saved for each class

    os.makedirs(save_dir, exist_ok=True)  # Create directory to save images

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Save one image per class showing the actual and predicted labels
            for i in range(len(inputs)):
                actual_class = class_names[labels[i].cpu().numpy()]
                predicted_class = class_names[preds[i].cpu().numpy()]

                # Save the image if not already saved for this class
                if not class_image_saved[actual_class]:
                    img = inputs[i].cpu().numpy().transpose((1, 2, 0))  # Convert image back to HWC format
                    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
                    img = np.clip(img, 0, 1)  # Clip values to valid range [0, 1]

                    # Plot and save the image with the title showing predicted and actual labels
                    plt.imshow(img)
                    plt.title(f"Actual: {actual_class}, Predicted: {predicted_class}")
                    plt.axis('off')  # Hide axes
                    save_path = os.path.join(save_dir, f'{actual_class}_predicted_as_{predicted_class}.png')
                    plt.savefig(save_path)
                    plt.close()  # Close the figure
                    class_image_saved[actual_class] = True  # Mark the class as saved

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Compute classification metrics
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Additional metrics: accuracy, precision, recall, f1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')  # Average over all classes
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

# Assuming you have a DataLoader for your test dataset
# Example:
test_dir = 'test'
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Define class names
class_names = test_dataset.classes

# Load the checkpoint first
checkpoint_path = '/home/mzrdu/Downloads/SwinTransformer/checkpoints/model.pth'
checkpoint = torch.load(checkpoint_path)

# Create an instance of the model
model = Swin_T(num_classes=len(test_dataset.classes))

# Load the model weights from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Move the model to the appropriate device (e.g., GPU or CPU)
model = model.to(device)

# Call the function to compute metrics and save predicted vs actual images
compute_metrics_and_save_images(model, test_loader, device, class_names)
