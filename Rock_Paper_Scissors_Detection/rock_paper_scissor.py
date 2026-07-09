import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])



train_dataset = datasets.ImageFolder(
    root="rock_paper_scissors_train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root="rock_paper_scisscors_test",
    transform=test_transform
)

print("Classes:", train_dataset.classes)
print("Training Images:", len(train_dataset))
print("Testing Images:", len(test_dataset))

num_classes = len(train_dataset.classes)



train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128,num_classes)

        )

    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x



model = CNN().to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)



scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


epochs = 15

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")


model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")



torch.save(
    model.state_dict(),
    "rock_paper_scissors_model.pth"
)

print("Model Saved!")

#SINGLE IMAGE PREDICTION

image_path = "rock_paper_scisscors_test/paper/testpaper01-00.png"

image = Image.open(
    image_path
).convert("RGB")

image = test_transform(image)

image = image.unsqueeze(0)

image = image.to(device)

model.eval()

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

if prediction == 0:
    print("\nPrediction: PAPER")

elif prediction == 1:
    print("\nPrediction: ROCK")

else:
    print("\nPrediction: SCISSORS")
