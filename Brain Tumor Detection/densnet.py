import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

transforming_img = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_path = ('./Brain-Tumor-Classification-DataSet/Training/')
test_path = ('./Brain-Tumor-Classification-DataSet/Testing/')
val_path = ('./Brain-Tumor-Classification-DataSet/val/')
train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transforming_img),
    batch_size=64, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transforming_img),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
        torchvision.datasets.ImageFolder(val_path, transform=transforming_img),
        batch_size=64, shuffle=True
)

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])




train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))
val_count=len(glob.glob(val_path+'/**/*.jpg'))




device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.densenet161(pretrained=True)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 4)
model = model.to(device)



optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
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
        torch.save(model.state_dict(),'./model/densenet_model.pth')
        best_accuracy=val_accuracy
model.load_state_dict(torch.load('./model/densenet_model.pth', map_location="cpu"))
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