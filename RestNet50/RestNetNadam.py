import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom NAdam implementation
class NAdam(Adam):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

num_classes = 10
batch_size = 128
num_epochs = 50
learning_rate = 0.001  # Learning rate for adaptive methods
betas = (0.9, 0.999)  # Nadam coefficients

# For MNIST, we need to convert single channel to 3 channels for ResNet
class GrayscaleToRGB(nn.Module):
    def forward(self, x):
        return x.repeat(1, 3, 1, 1)

transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.Resize(224),  # ResNet50 requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
    GrayscaleToRGB()
])

transform_test = transforms.Compose([
    transforms.Resize(224),  # ResNet50 requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
    GrayscaleToRGB()
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

model = models.resnet50(weights="IMAGENET1K_V2")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = NAdam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=5e-4)

train_loss_list = []
accuracy_list = []

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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(train_loss_list, label='Nadam')
plt.xlabel("Epochs")
plt.ylabel("Training loss")
plt.title("MNIST Training Loss for Nadam Optimizer")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(accuracy_list, label='Nadam')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("MNIST Training Accuracy for Nadam Optimizer")
plt.legend(loc='lower right')
plt.show()