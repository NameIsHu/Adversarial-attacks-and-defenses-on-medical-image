import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])



train_path = ('./dataset/train/')
test_path = ('./dataset/test/')
val_path = ('./dataset/val/')
train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=train_transform),
    batch_size=128, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=test_transform),
    batch_size=128, shuffle=False
)
val_loader = DataLoader(
        torchvision.datasets.ImageFolder(val_path, transform=test_transform),
        batch_size=128, shuffle=True
)

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])




train_count=len(glob.glob(train_path+'/**/*.png'))
test_count=len(glob.glob(test_path+'/**/*.png'))
val_count=len(glob.glob(val_path+'/**/*.png'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(in_features=512, out_features=5)
model = model.to(device)


optimizer = Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

best_accuracy = 0.0
os.makedirs('./model', exist_ok=True)

for epoch in range(20):


    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    model.eval()
    
    val_accuracy = 0.0
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        val_accuracy += int(torch.sum(prediction == labels.data))

    val_accuracy = val_accuracy / val_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy))
    if val_accuracy>best_accuracy:
        torch.save(model.state_dict(),'./model/resnet2.pth')
        best_accuracy=val_accuracy
model.load_state_dict(torch.load('./model/resnet2.pth', map_location="cpu"))
model = model.to(device)
test_accuracy=0.0
for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
test_accuracy = test_accuracy / test_count
print('Test Accuracy: '+str(test_accuracy))