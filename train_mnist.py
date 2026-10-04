import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# =========================
# 1. Basic settings
# =========================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("Device:", device)
print("PyTorch version:", torch.__version__)
print("=" * 60)


# =========================
# 2. CNN model
# =========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 3. Load MNIST
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

data_dir = "./data"

train_dataset = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=0
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# =========================
# 4. Create model
# =========================
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nModel:")
print(model)


# =========================
# 5. Train
# =========================
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(1, EPOCHS + 1):

    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -------------------------
    # Test
    # -------------------------
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    test_losses.append(avg_test_loss)
    test_accuracies.append(accuracy)

    print(
        f"Epoch [{epoch}/{EPOCHS}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Test Accuracy: {accuracy:.2f}%"
    )


# =========================
# 6. Save model
# =========================
os.makedirs("models", exist_ok=True)

model_path = "models/mnist_cnn.pth"
torch.save(model.state_dict(), model_path)

print("\nModel saved to:", model_path)


# =========================
# 7. Plot training curves
# =========================
os.makedirs("results", exist_ok=True)

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MNIST Training and Test Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=150)
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS + 1), test_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("MNIST Test Accuracy")
plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=150)
plt.close()

print("Figures saved to results/")
print("\nDAY 1 COMPLETE!")
