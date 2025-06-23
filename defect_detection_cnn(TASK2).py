# 📚 Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import Counter

# 📌 Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 📦 Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 📥 Load Full Dataset
full_data = datasets.ImageFolder(root='data/casting_data', transform=transform)
print(f'Full dataset class distribution: {Counter(full_data.targets)}')

# 📦 Split into Train & Test (80% Train, 20% Test)
train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = random_split(full_data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 📊 Check class distribution
train_targets = [full_data.targets[i] for i in train_data.indices]
test_targets = [full_data.targets[i] for i in test_data.indices]
print("Train class distribution:", Counter(train_targets))
print("Test class distribution:", Counter(test_targets))

# 📌 Define CNN Model
class DefectCNN(nn.Module):
    def __init__(self):
        super(DefectCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 30 * 30)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 📌 Initialize Model, Loss, Optimizer
model = DefectCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📊 Training the Model
epochs = 10
train_losses = []

for epoch in range(epochs):
    running_loss = 0.0
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# 📈 Plot Training Loss
plt.plot(train_losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 📊 Evaluate Model on Test Set
correct = 0
total = 0
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f'\n✅ Test Accuracy: {accuracy:.2f}%')

# 📊 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_data.classes)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 📊 Classification Report
print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=full_data.classes))

# 📸 Visualize Sample Predictions
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Get one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Show 5 Sample Predictions with readable labels
for idx in range(5):
    pred_label = full_data.classes[preds[idx]]
    actual_label = full_data.classes[labels[idx]]

    # Custom readable message based on prediction
    if pred_label == 'ok_front':
        pred_message = "✅ Good Product"
    else:
        pred_message = "⚠️ Defect Detected — Review Required"

    # Check if prediction is correct
    if pred_label == actual_label:
        correctness = "Correct Prediction"
    else:
        correctness = f"Incorrect (Actual: {'Good Product' if actual_label=='ok_front' else 'Defective'})"

    # Display image and message
    imshow(images[idx].cpu(), title=f'{pred_message}\n{correctness}')
