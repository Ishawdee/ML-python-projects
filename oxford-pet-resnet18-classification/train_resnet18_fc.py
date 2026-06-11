# multi-class classification for 37 different pet breed, experiment 1
# using OxfordIIITPet dataset and ResNet18 pretrained model
# freezing all layers but the final fc layer
# 84.71% test accuracy

import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import OxfordIIITPet
from pathlib import Path

base_path = Path("C:/Users/shadi/Desktop/python stuff (new)/multi classification")
dataset_path = base_path / "data"

# downloading train, val, test data
trainval_data = OxfordIIITPet(
    root=dataset_path,
    split="trainval",
    target_types="category",
    download=True,
    transform=None
)

test_data = OxfordIIITPet(
    root=dataset_path,
    split="test",
    target_types="category",
    download=True,
    transform=None
)
# 3680 trainval samples, 3669 test

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# splitting train and validation sets
labels = [trainval_data[i][1] for i in range(len(trainval_data))]
indices = list(range(len(trainval_data)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# transforms
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# creating datasets
from torchvision.datasets import OxfordIIITPet

train_transform = OxfordIIITPet(
    root=dataset_path,
    split="trainval",
    target_types="category",
    download=False,
    transform=train_transform
)

val_transform = OxfordIIITPet(
    root=dataset_path,
    split="trainval",
    target_types="category",
    download=False,
    transform=val_test_transform
)

test_data = OxfordIIITPet(
    root=dataset_path,
    split="test",
    target_types="category",
    download=False,
    transform=val_test_transform
)
from torch.utils.data import Subset

train_data = Subset(train_transform, train_indices)
val_data = Subset(val_transform, val_indices)

# creating dataloaders
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))

"""print(images.shape)
print(labels.shape)
print(labels[:10])
print(len(train_data), len(val_data), len(test_data))
print(trainval_data.classes[:10])
print(len(trainval_data.classes))"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# pre-trained model loading
from torchvision.models import resnet18

model = resnet18(weights="DEFAULT")

# freezing early layers/disable gradient updates during training to prevent overfitting
for param in model.parameters():
    param.requires_grad = False

# modifying the final layer to match the current number of features
num_ftrs = model.fc.in_features # 512 features
model.fc = nn.Linear(in_features=num_ftrs, out_features=37) # 37 classes

model = model.to(device)

# loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.fc.parameters(), lr=0.001) # only the fc layer

# training on the final layer
from sklearn import metrics

DO_TRAIN = False # already trained
DO_TEST = True
num_epochs = 10
patience = 4
best_val_loss = float("inf")
epochs_without_improvement = 0
best_epoch = 0

if DO_TRAIN:
    for epoch in range(num_epochs):

        model.train()

        train_loss = 0
        train_true = []
        train_preds = []

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            pred_labels = preds.argmax(dim=1)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_true.extend(labels.cpu().numpy())
            train_preds.extend(pred_labels.detach().cpu().numpy())
        
        avg_train_loss = train_loss / len(train_data)
        train_acc = metrics.accuracy_score(train_true, train_preds)


        model.eval()

        val_loss = 0
        val_true = []
        val_preds = []


        with torch.inference_mode():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                val_outputs = model(images)
                val_labels = val_outputs.argmax(dim=1)
                loss = loss_fn(val_outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_true.extend(labels.cpu().numpy())
                val_preds.extend(val_labels.detach().cpu().numpy())


        avg_val_loss = val_loss / len(val_data)
        val_acc = metrics.accuracy_score(val_true, val_preds)

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_resnet18_pet.pth")
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

        torch.save(checkpoint, "latest_resnet18_pet_checkpoint.pth")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")


if DO_TEST:

    model.load_state_dict(torch.load("best_resnet18_pet.pth", map_location=device))

    test_loss = 0
    test_true = []
    test_preds = []

    model.eval()

    with torch.inference_mode():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            test_outputs = model(images)
            test_labels = test_outputs.argmax(dim=1)
            loss = loss_fn(test_outputs, labels)

            test_loss += loss.item() * images.size(0)
            test_preds.extend(test_labels.detach().cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_data)
    test_acc = metrics.accuracy_score(test_true, test_preds)

    print(f"Test loss: {avg_test_loss:.4f}")
    print(f"Test acc: {test_acc:.4f}")


    # Evaluation
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(test_true, test_preds)
    class_names = test_data.classes

    # Copy confusion matrix, remove diagonal
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    # top confused pairs
    num_pairs = 10
    flat_indices = np.argsort(cm_no_diag.ravel())[::-1]

    print("Top confused pairs:")

    count = 0
    for idx in flat_indices:
        true_idx, pred_idx = np.unravel_index(idx, cm_no_diag.shape)
        value = cm_no_diag[true_idx, pred_idx]

        if value == 0:
            break

        print(
            f"True: {class_names[true_idx]} | "
            f"Predicted: {class_names[pred_idx]} | "
            f"Count: {value}"
        )

        count += 1
        if count == num_pairs:
            break
