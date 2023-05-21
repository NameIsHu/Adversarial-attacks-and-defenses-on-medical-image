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
    train_path = ('./dataset/train/')
    test_path = ('./dataset/test/')
    val_path = ('./dataset/val/')

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transforming_img),
        batch_size=128, shuffle=True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transforming_img),
        batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        torchvision.datasets.ImageFolder(val_path, transform=transforming_img),
        batch_size=128, shuffle=True
    )

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=5)
    model.load_state_dict(torch.load('./model/resnet2.pth', map_location="cpu"))
    device = torch.device('cuda')
    model = model.to(device)
    print("undercover attack")
    print("use resnet model for Diabetic Retinopathy")
    attacks = {}
    pgd = torchattacks.PGD(model=model, eps=0.1, steps=10)
    fgsm = torchattacks.FGSM(model=model, eps=0.1)
    cw = torchattacks.CW(model=model)
    bim = torchattacks.BIM(model=model)
    deepFool = torchattacks.DeepFool(model=model)
    onePixel = torchattacks.OnePixel(model=model)


    attacks["OnePixel"] = onePixel
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    for attack_name, attack in attacks.items():
        print("use",attack_name,"adversarial training")
        best_acc=0
        model.load_state_dict(torch.load('./model/resnet2.pth', map_location="cpu"))
        model = model.to(device)
        for epoch in range(40):

            train_loss=0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                model.eval()
                adv_images = attack(images, labels)
                model.train()
                optimizer.zero_grad()
                outputs = model(images)
                adv_out =model(adv_images)
                loss1 = loss_function(outputs, labels)
                loss2=loss_function(adv_out, labels)
                loss=loss1+loss2*0.8
                
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
                torch.save(model.state_dict(), './model/1undercover_resnet_model'+attack_name+'.pth')
            torch.save(model.state_dict(), './model/undercover_resnet_model'+attack_name+'.pth')
        model.load_state_dict(torch.load('./model/1undercover_resnet_model'+attack_name+'.pth', map_location="cpu"))
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
