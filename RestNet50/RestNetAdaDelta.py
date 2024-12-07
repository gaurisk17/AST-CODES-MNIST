import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

num_classes = 10
batch_size = 128
num_epochs = 50
rho = 0.9  # Decay factor for AdaDelta
eps = 1e-6  # Small constant for numerical stability

# Convert grayscale to RGB
class GrayscaleToRGB(nn.Module):
    def forward(self, x):
        return x.repeat(1, 3, 1, 1)

# Wrapper for ResNet to handle grayscale input
class MNISTResNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(MNISTResNet, self).__init__()
        self.grayscale_to_rgb = GrayscaleToRGB()
        self.resnet = original_model
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.grayscale_to_rgb(x)
        return self.resnet(x)

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.Resize(224),  # ResNet50 requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False)

# Initialize model
base_model = models.resnet50(weights="IMAGENET1K_V2")
model = MNISTResNet(base_model, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# AdaDelta optimizer
optimizer = optim.Adadelta(model.parameters(), rho=rho, eps=eps,
                          weight_decay=5e-4)

# Training metrics storage
train_loss_list = []
accuracy_list = []

# Training loop
print("Training with AdaDelta optimizer")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    train_loss_list.append(train_loss)
    accuracy_list.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, "
          f"Accuracy: {accuracy:.2f}%")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss_list, label='AdaDelta')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("MNIST Training Loss with AdaDelta Optimizer")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(accuracy_list, label='AdaDelta')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("MNIST Training Accuracy with AdaDelta Optimizer")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Evaluation on test set
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Testing"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

test_accuracy = 100. * test_correct / test_total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")
