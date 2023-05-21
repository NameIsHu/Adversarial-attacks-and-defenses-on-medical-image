import numpy as np
import random
import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchattacks
from torchvision.utils import save_image
from torch.optim import Adam
if __name__ == '__main__':
    transforming_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_path = ('./COVID-19_Radiography_Dataset/train/')
    test_path = ('./COVID-19_Radiography_Dataset/test/')
    val_path = ('./COVID-19_Radiography_Dataset/val/')

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transforming_img),
        batch_size=384, shuffle=True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transforming_img),
        batch_size=384, shuffle=True
    )
    val_loader = DataLoader(
        torchvision.datasets.ImageFolder(val_path, transform=transforming_img),
        batch_size=384, shuffle=True
    )
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=3)
    model.load_state_dict(torch.load('./model/model_noNorm.pth', map_location="cpu"))
    device = torch.device('cuda')
    model = model.to(device)
    print("use resnet model for covid-19")
    attacks = {}
    pgd = torchattacks.PGD(model=model, eps=0.1, steps=10)
    fgsm = torchattacks.FGSM(model=model, eps=0.1)
    cw = torchattacks.CW(model=model)
    bim = torchattacks.BIM(model=model)
    deepFool = torchattacks.DeepFool(model=model)
    onePixel = torchattacks.OnePixel(model=model)

    attacks["PGD"] = pgd
    attacks["FGSM"] = fgsm
    attacks["CW"] = cw
    attacks["BIM"] = bim
    attacks["DeepFool"] = deepFool
    attacks["OnePixel"] = onePixel
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    for attack_name, attack in attacks.items():
        print("use",attack_name,"adversarial training")
        model.load_state_dict(torch.load('./model/model_noNorm.pth', map_location="cpu"))
        model = model.to(device)
        best_acc=0
        for epoch in range(40):

            train_loss=0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                p = np.random.uniform(0.0, 1.0)
                if p > 0.5:
                    model.eval()
                    images = attack(images, labels)
                    images=images.detach()
                model.train()
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.data * images.size(0)
            print("train loss",train_loss)
            total = 0
            correct = 0
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                adv_pred = torch.max(outputs, 1)
                correct += int((adv_pred.indices == labels).sum())
                total += len(outputs)
            if correct/total>=best_acc:
                best_acc=correct/total
                torch.save(model.state_dict(), './model/1resnet_model'+attack_name+'.pth')
            torch.save(model.state_dict(), './model/resnet_model'+attack_name+'.pth')
        total = 0
        correct = 0
        model.load_state_dict(torch.load('./model/1resnet_model'+attack_name+'.pth', map_location="cpu"))
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            adv_pred = torch.max(outputs, 1)
            correct += int((adv_pred.indices == labels).sum())
            total += len(outputs)
            print("original acc",correct/total)

        total = 0
        correct = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            images = attack(images, labels)
            outputs = model(images)
            adv_pred = torch.max(outputs, 1)
            correct += int((adv_pred.indices == labels).sum())
            total += len(outputs)
        print(attack_name,"acc",correct/total)
