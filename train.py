import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from google.colab import drive
drive.mount('/content/drive')

import torchvision.transforms as transforms
import torch
import os
import random
import zipfile
import shutil
from PIL import Image

# ============================
# SETTINGS
# ============================
ZIP_FILE = "/content/drive/My Drive/profile_235.zip"  # zipped profiles folder in Drive
EXTRACT_FOLDER = "./profiles"                   
TRAIN_FOLDER = "./profiles_train"
TEST_FOLDER = "./profiles_test"
TRAIN_RATIO = 0.8  # 80% train, 20% test
batch_size = 64

# ============================
# UNZIP PROFILES
# ============================
if not os.path.exists(EXTRACT_FOLDER):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    print(f"Extracted {ZIP_FILE} to {EXTRACT_FOLDER}")
else:
    print(f"{EXTRACT_FOLDER} already exists, skipping extraction.")

# ============================
# CREATE TRAIN / TEST FOLDERS
# ============================
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# ============================
# GET ALL PROFILE FOLDERS
# ============================
all_profiles = [f for f in os.listdir(EXTRACT_FOLDER)
                if os.path.isdir(os.path.join(EXTRACT_FOLDER, f)) and not f.startswith(".")]

# shuffle
random.seed(42)  
random.shuffle(all_profiles)

# split
train_cutoff = int(len(all_profiles) * TRAIN_RATIO)
train_profiles = all_profiles[:train_cutoff]
test_profiles = all_profiles[train_cutoff:]

# ============================
# MOVE TO TRAIN / TEST FOLDERS
# ============================
def move_profiles(profile_list, dest_folder):
    for profile in profile_list:
        src_path = os.path.join(EXTRACT_FOLDER, profile)
        dest_path = os.path.join(dest_folder, profile)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(src_path, dest_path)

move_profiles(train_profiles, TRAIN_FOLDER)
move_profiles(test_profiles, TEST_FOLDER)

print(f"Total profiles: {len(all_profiles)}")
print(f"Train profiles: {len(train_profiles)}")
print(f"Test profiles: {len(test_profiles)}")

# ============================
# Training Transform (with augmentation)
# ============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1,
                           contrast=0.1,
                           saturation=0.1,
                           hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================
# Test Transform (deterministic)
# ============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

############################################
# CUSTOM DATASET FOR 6 IMAGES + LABEL
############################################
class ProfileDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.folders = []

        for f in sorted(os.listdir(root)):
            folder_path = os.path.join(root, f)
            if not os.path.isdir(folder_path) or f.startswith("."):
                continue

            # find all jpg/png images in folder
            img_files = [fname for fname in os.listdir(folder_path)
                         if fname.lower().endswith((".jpg", ".png"))]

            if len(img_files) < 6:
                continue

            # must have label.txt or rating.txt
            if not (os.path.exists(os.path.join(folder_path, "label.txt")) or
                    os.path.exists(os.path.join(folder_path, "rating.txt"))):
                continue

            self.folders.append(f)

        print("Folders found:", self.folders)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        path = os.path.join(self.root, folder)

        # get first 6 images
        img_files = sorted([fname for fname in os.listdir(path)
                            if fname.lower().endswith((".jpg", ".png"))])[:6]

        images = []
        for img_file in img_files:
            img_path = os.path.join(path, img_file)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)

        # read label 
        label = None
        for file_name in ["label.txt", "rating.txt"]:
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    content = f.read().strip().lower().replace("\n", "").replace("\r", "")
                    if content in ["like", "1"]:
                        label = 1
                    elif content in ["dislike", "0"]:
                        label = 0
                    else:
                        try:
                            label = int(content)
                        except ValueError:
                            raise ValueError(f"Invalid label in {file_path}: {repr(content)}")
                break

        if label is None:
            raise ValueError(f"No label or rating found for profile {folder}")

        return images, torch.tensor(label, dtype=torch.long)



# ============================
# CREATE DATALOADERS
# ============================
trainset = ProfileDataset(root=TRAIN_FOLDER, transform=train_transform)


testset = ProfileDataset(root=TEST_FOLDER, transform=test_transform)


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("Dataset and DataLoaders ready!")


def __getitem__(self, idx):
    folder = self.folders[idx]
    path = os.path.join(self.root, folder)

    # sort images to keep order consistent
    img_files = sorted([fname for fname in os.listdir(path)
                        if fname.lower().endswith((".jpg", ".png"))])[:6]

    images = []
    for img_file in img_files:
        img_path = os.path.join(path, img_file)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        images.append(img)

    images = torch.stack(images)

    # load label from label.txt or rating.txt
    label = None
    for file_name in ["label.txt", "rating.txt"]:
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read().strip().lower()
                content = content.replace("\n", "").replace("\r", "").strip()

                # map text to integer
                if content in ["like", "1"]:
                    label = 1
                elif content in ["dislike", "0"]:
                    label = 0
                else:
                    # try parsing int
                    try:
                        label = int(content)
                    except ValueError:
                        raise ValueError(f"Invalid label in {file_path}: {repr(content)}")
            break

    if label is None:
        raise ValueError(f"No label or rating found for profile {folder}")

    return images, torch.tensor(label, dtype=torch.long)


from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()   # remove ImageNet classifier

        # Small classifier head (only thing being trained)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x shape: (B, 6, 3, 224, 224)
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)         # flatten profile images
        feats = self.backbone(x)          # (B*N, 512)

        feats = feats.view(B, N, 512)     # regroup into profiles
        feats = feats.mean(dim=1)         # average 6 embeddings

        out = self.classifier(feats)
        return out



net = Net()

for param in net.backbone.parameters():
    param.requires_grad = False
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.classifier.parameters(), lr=0.001)




import torch
import copy
import math

# parameters
num_epochs = 2000
patience = 50
min_delta = 0.0
save_path = 'best_profile_net.pth'
monitor = 'val_acc'  

best_metric = -math.inf      
best_epoch = -1
epochs_no_improve = 0

# helper to evaluate on validation/test set
def evaluate(net, dataloader, device='cpu'):
    net.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(trainset)  
    val_loss, val_acc = evaluate(net, testloader, device) 

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

 
    if val_acc - best_metric > min_delta:
        best_metric = val_acc
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(net.state_dict(), save_path)
        print(f"  New best val_acc={best_metric:.4f} -> saved {save_path}")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} with val_acc={best_metric:.4f}")
        break

# after training, load best weights
net.load_state_dict(torch.load(save_path))
net.to(device)
print("Loaded best model from", save_path)


You can save your trained model using:

PATH = './profile_net (1).pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(total)
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ============================
# Grad Cam Visualization of training profile
# ============================
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ===== 1. GET A RANDOM PROFILE FROM TRAINLOADER =====
images, labels = next(iter(trainloader))   # [B, 6, C, H, W]
profile = images[0].unsqueeze(0)           # [1, 6, C, H, W]

net.eval()

# ===== 2. AUTOMATICALLY FIND LAST CONV LAYER =====
target_layer = None
for name, layer in net.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        target_layer = layer

print("Using Grad-CAM layer:", target_layer)

# ===== 3. HOOKS =====
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ===== 4. FORWARD + BACKWARD PASS =====
output = net(profile)
pred_class = output.argmax(dim=1)

net.zero_grad()
output[0, pred_class].backward()

# ===== 5. GRAD-CAM COMPUTATION =====
grads = gradients[0].detach()
activations = feature_maps[0].detach()

cams = []
for i in range(activations.shape[0]):
    weights = grads[i].mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * activations[i]).sum(dim=0))

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    cam_resized = cv2.resize(
        cam.cpu().numpy(),
        (profile.shape[4], profile.shape[3])
    )
    cams.append(cam_resized)

# ===== 6. DISPLAY ALL 6 IMAGES WITH HEATMAPS =====
for i in range(6):
    img_np = profile[0, i].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.imshow(cams[i], cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f"Image {i} | Pred = {pred_class.item()}")
    plt.show()


# ============================
# Grad Cam Visualization of Test Profile
# ============================
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

# ===== 1. GET ALL TEST DATA =====
all_test_images = []
all_test_labels = []

for batch_images, batch_labels in testloader:
    all_test_images.append(batch_images)
    all_test_labels.append(batch_labels)

all_test_images = torch.cat(all_test_images, dim=0)  # [num_profiles, 6, C, H, W]
all_test_labels = torch.cat(all_test_labels, dim=0)

# ===== 2. PICK RANDOM TEST PROFILE =====
profile_idx = random.randint(0, all_test_images.shape[0] - 1)
profile = all_test_images[profile_idx].unsqueeze(0)  # [1, 6, C, H, W]

net.eval()

# ===== 3. AUTOMATICALLY FIND LAST CONV LAYER =====
target_layer = None
for name, layer in net.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        target_layer = layer

print("Using Grad-CAM layer:", target_layer)

# ===== 4. HOOKS =====
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ===== 5. FORWARD + BACKWARD PASS =====
output = net(profile)
pred_class = output.argmax(dim=1)

net.zero_grad()
output[0, pred_class].backward()

# ===== 6. GRAD-CAM COMPUTATION =====
grads = gradients[0].detach()
activations = feature_maps[0].detach()

cams = []
for i in range(activations.shape[0]):
    weights = grads[i].mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * activations[i]).sum(dim=0))

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    cam_resized = cv2.resize(
        cam.cpu().numpy(),
        (profile.shape[4], profile.shape[3])
    )
    cams.append(cam_resized)

# ===== 7. DISPLAY ALL 6 IMAGES WITH HEATMAPS =====
for i in range(6):
    img_np = profile[0, i].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.imshow(cams[i], cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f"Profile {profile_idx} | Image {i} | Pred = {pred_class.item()}")
    plt.show()


# ============================
# Display confusion matrix
# ============================

# 1a. Print counts per label
from collections import Counter
train_labels = []
for _, lbl in trainset:
    train_labels.append(int(lbl))
print("Train label counts:", Counter(train_labels))

test_labels = []
for _, lbl in testset:
    test_labels.append(int(lbl))
print("Test label counts:", Counter(test_labels))

# 1b. Compute train / test accuracy (explicit)
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    preds = []
    trues = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            preds.extend(predicted.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    acc = 100 * correct / total if total>0 else 0
    return acc, preds, trues

train_acc, _, _ = evaluate(trainloader, net)
test_acc, test_preds, test_trues = evaluate(testloader, net)
print("Train acc:", train_acc, " Test acc:", test_acc)

# 1c. Confusion matrix (simple)
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion matrix (test):")
print(confusion_matrix(test_trues, test_preds))
print(classification_report(test_trues, test_preds))
