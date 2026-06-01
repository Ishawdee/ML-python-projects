# TASK: training a binary classification model on 2000 images of cats and dogs using CNN

from pathlib import Path
# first, we see how many samples we have, and how they're named!
base_path = Path("C:/Users/shadi/Desktop/python stuff (new)/binary classification")

cat_path = base_path / "Cat"
dog_path = base_path / "Dog"

cat_files = [file.name for file in cat_path.iterdir() if file.is_file()]
dog_files = [file.name for file in dog_path.iterdir() if file.is_file()]

# print(len(cat_files))
# print(len(dog_files))
# these give us 12501 samples each.

# =========================

# shuffling and splitting train/test/val
import random

cat_files = sorted(cat_files)
dog_files = sorted(dog_files)

def shuffle(cats, dogs):
    x = cats.copy()
    random.seed(42)
    random.shuffle(cats)
    random.shuffle(dogs)

    y = cats.copy()
    # print("before shuffle:", x[:10], "\nafter shuffle:", y[:10])
    # print(x == y) it's a false; a proof that our data samples are shuffled (the same
    # goes for dogs)

    return cats, dogs

cat_files, dog_files = shuffle(cat_files, dog_files)

train_cat = cat_files[:1000]
val_cat = cat_files[1000:1200]
test_cat = cat_files[1200:1400]

train_dog = dog_files[:1000]
val_dog = dog_files[1000:1200]
test_dog = dog_files[1200:1400]

# =========================

import shutil

def copy_images(file_list, source_folder, destination_folder):
    destination_folder.mkdir(parents=True, exist_ok=True)
    for file_name in file_list:
        source_path = source_folder / file_name
        destination_path = destination_folder / file_name
        shutil.copy2(source_path, destination_path)

copy_images(train_cat, cat_path, base_path / "data/train/cat")
copy_images(val_cat, cat_path, base_path / "data/val/cat")
copy_images(test_cat, cat_path, base_path / "data/test/cat")

copy_images(train_dog, dog_path, base_path / "data/train/dog")
copy_images(val_dog, dog_path, base_path / "data/val/dog")
copy_images(test_dog, dog_path, base_path / "data/test/dog")

folders_to_check = [
    base_path / "data/train/cat",
    base_path / "data/train/dog",
    base_path / "data/val/cat",
    base_path / "data/val/dog",
    base_path / "data/test/cat",
    base_path / "data/test/dog",
]

#for folder in folders_to_check:
#    print(folder, len(list(folder.iterdir())))

# =========================
# image to tensor, creating datasets and dataloaders

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_transform = transforms.Compose([ # data augmentation only for training
    transforms.Resize((150, 150)),
    transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dir = base_path / "data/train"
val_dir = base_path / "data/val"
test_dir = base_path / "data/test"

# creating datasets (containing tensors)
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_data = datasets.ImageFolder(test_dir, transform=val_test_transform)

# print(train_data.classes) --> cat and dog
# print(train_data.class_to_idx) -->indices of cat and dog
# print(len(train_data), len(val_data), len(test_data)) --> samples count

# creating dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))

# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # original feature layers: [32, 3, 128, 128]
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(), # [32, 16, 128, 128]
            nn.MaxPool2d(2), # [32, 16, 64, 64]

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [32, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2), # [32, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [32, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2), # [32, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [32, 128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2), # [32, 128, 8, 8]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


"""model = CatDogCNN()
images, labels = next(iter(train_loader))
outputs = model(images)
print(outputs.shape) #torch.Size([32, 2])
# 32 = number of images in this batch, 2  = number of output scores per image"""

model = CatDogCNN().to(device)
model.load_state_dict(
    torch.load("best_cat_dog_cnn.pth", map_location=device)
)

# =========================
# loss & optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
DO_TRAIN = False
DO_TEST = True

# training loop

from sklearn import metrics

best_val_loss = 0.4951
best_epoch = 34
start_epoch = 34
epochs = 60
#best_epoch = 0
patience = 8
epochs_without_improvement = 0
#best_val_loss = float("inf")

if DO_TRAIN:
    for epoch in range(start_epoch, epochs):

        model.train()

        train_loss = 0
        train_true = []
        train_pred = []

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            
            preds = model(images) # returns logits of shape [32, 2]
            pred_labels = preds.argmax(dim=1) # returns class labels
            loss = loss_fn(preds, labels) # CrossEntropy needs raw logits
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(pred_labels.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_data)
        train_accuracy = metrics.accuracy_score(train_true, train_pred)

        # =====================
        # Validation
        # =====================

        model.eval()

        val_loss = 0
        val_true = []
        val_pred = []

        with torch.inference_mode():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)
                val_preds = model(images)
                val_preds_labels = val_preds.argmax(dim=1)
                loss = loss_fn(val_preds, labels)
                val_loss += loss.item() * images.size(0)

                val_true.extend(labels.cpu().numpy())
                val_pred.extend(val_preds_labels.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_data)
        val_accuracy = metrics.accuracy_score(val_true, val_pred)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_cat_dog_cnn.pth")
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

        torch.save(checkpoint, "latest_checkpoint.pth")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # =====================
        # Print results
        # =====================
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train loss: {avg_train_loss:.4f} | "
                f"Train accuracy: {train_accuracy:.4f} | "
                f"Val loss: {avg_val_loss:.4f} | "
                f"Val accuracy: {val_accuracy:.4f}"
                )
    print(f"Best model was saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")

# =====================
# testing

if DO_TEST:
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load("best_cat_dog_cnn.pth", map_location=device))
    model.eval()

    test_loss = 0
    test_true = []
    test_pred = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            test_preds = model(images)
            loss = loss_fn(test_preds, labels)
            test_preds_labels = test_preds.argmax(dim=1)

            test_loss += loss.item() * images.size(0)
            test_true.extend(labels.cpu().numpy())
            test_pred.extend(test_preds_labels.detach().cpu().numpy())

    avg_test_loss = test_loss / len(test_data)
    test_accuracy = metrics.accuracy_score(test_true, test_pred)

    print(f"Test loss: {avg_test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # =====================
    # Confusion matrix
    # =====================
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    from PIL import Image


    class_names = test_data.classes  # ['cat', 'dog']

    cm = confusion_matrix(test_true, test_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot()
    plt.title("Confusion Matrix - Test Set")
    plt.show()

    print(classification_report(
        test_true,
        test_pred,
        target_names=class_names
    ))

    # =====================
    # Find wrong predictions
    # =====================

    wrong_indices = []

    for i, (true_label, predicted_label) in enumerate(zip(test_true, test_pred)):
        if true_label != predicted_label:
            wrong_indices.append(i)

    print(f"Number of wrong predictions: {len(wrong_indices)}")
    print(f"Number of correct predictions: {len(test_data) - len(wrong_indices)}")
