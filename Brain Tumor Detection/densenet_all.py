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

if __name__ == '__main__':
    transforming_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_path = ('./Brain-Tumor-Classification-DataSet/Training/')
    test_path = ('./Brain-Tumor-Classification-DataSet/Testing/')

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transforming_img),
        batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transforming_img),
        batch_size=32, shuffle=True
    )

    model = torchvision.models.densenet161(pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 4)
    model.load_state_dict(torch.load('./model/densenet_model.pth', map_location="cpu"))
    device = torch.device('cuda')
    cpu_device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    print("use densenet model for Brain Tumor Classification")
    train_count = 0
    train_accuracy = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        train_count += len(outputs)
        train_accuracy += int(torch.sum(prediction == labels.data))
    train_accuracy = train_accuracy / train_count
    test_accuracy = 0.0
    test_count = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
        test_count += len(outputs)
    test_accuracy = test_accuracy / test_count
    images=images.to(cpu_device)
    labels=labels.to(cpu_device)
    outputs=outputs.to(cpu_device)
    print("train num {} , original accuracy{} ,test num {} ,original accuracy{}".format(train_count, train_accuracy,
                                                                                        test_count, test_accuracy))
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

    for attack_name, attack in attacks.items():
        total = 0
        correct = 0
        try:
            for i, (atk_images, atk_labels) in enumerate(train_loader):
                atk_images = atk_images.to(device)
                atk_labels = atk_labels.to(device)

                x_adv = attack(atk_images, atk_labels)
                adv_op = model(x_adv)
                _, p_adv = torch.max(adv_op, dim=1)
                adv_pred = torch.max(adv_op, 1)
                correct += int((adv_pred.indices == atk_labels).sum())
                total += len(adv_op)
            print("train data num:", total, "Train Accuracy after ", attack_name, " attack:", (int(correct) / total))
            total = 0
            correct = 0
            for i, (atk_images, atk_labels) in enumerate(test_loader):
                atk_images = atk_images.to(device)
                atk_labels = atk_labels.to(device)

                x_adv = attack(atk_images, atk_labels)
                adv_op = model(x_adv)
                _, p_adv = torch.max(adv_op, dim=1)
                adv_pred = torch.max(adv_op, 1)
                correct += int((adv_pred.indices == atk_labels).sum())
                total += len(adv_op)
            print("test data num:", total, "Test Accuracy after ", attack_name, " attack:", (int(correct) / total))
        except:
            pass
