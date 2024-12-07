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
learning_rate = 0.002  # Lower learning rate for AdaMax
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# Convert grayscale to RGB
class GrayscaleToRGB(nn.Module):
    def forward(self, x):
        return x.repeat(1, 3, 1, 1)

# Wrapper for AlexNet to handle grayscale input
class MNISTAlexNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(MNISTAlexNet, self).__init__()
        self.grayscale_to_rgb = GrayscaleToRGB()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.grayscale_to_rgb(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.Resize(224),  # AlexNet requires 224x224 input
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
base_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
model = MNISTAlexNet(base_model, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# AdaMax optimizer
optimizer = optim.Adamax(model.parameters(), lr=learning_rate, 
                        betas=(beta1, beta2), eps=eps,
                        weight_decay=5e-4)

# Training metrics storage
train_loss_list = []
accuracy_list = []

# Training loop
print("Training AlexNet with AdaMax optimizer")
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
plt.plot(train_loss_list, label='AdaMax')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("MNIST Training Loss with AlexNet (AdaMax)")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(accuracy_list, label='AdaMax')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("MNIST Training Accuracy with AlexNet (AdaMax)")
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
