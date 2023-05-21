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

if __name__ == '__main__':
    transforming_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_path = ('./Brain-Tumor-Classification-DataSet/Training/')
    test_path = ('./Brain-Tumor-Classification-DataSet/Testing/')

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transforming_img),
        batch_size=128, shuffle=True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transforming_img),
        batch_size=128, shuffle=True
    )

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=4)
    model.load_state_dict(torch.load('./model/resnet_model.pth', map_location="cpu"))
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    print("use resnet model for Brain Tumor Classification")
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
    os.makedirs('./Adversarial Image/resnet', exist_ok=True)

    for attack_name, attack in attacks.items():
        for class1 in train_loader.dataset.classes:
            os.makedirs("./Adversarial Image/resnet/" + attack_name + "/training/" + class1, exist_ok=True)
            os.makedirs("./Adversarial Image/resnet/" + attack_name + "/testing/" + class1, exist_ok=True)
        count = 0
        for i, (atk_images, atk_labels) in enumerate(train_loader):
            atk_images = atk_images.to(device)
            atk_labels = atk_labels.to(device)
            x_adv = attack(atk_images, atk_labels)
            for img, y, in zip(x_adv, atk_labels):
                save_image(img,
                           "./Adversarial Image/resnet/" + attack_name + "/training/" + train_loader.dataset.classes[
                               int(y)] + "/" + str(count) + ".png")
                count+=1

        for i, (atk_images, atk_labels) in enumerate(test_loader):
            atk_images = atk_images.to(device)
            atk_labels = atk_labels.to(device)
            x_adv = attack(atk_images, atk_labels)
            for img, y, in zip(x_adv, atk_labels):
                save_image(img,
                           "./Adversarial Image/resnet/" + attack_name + "/testing/" + train_loader.dataset.classes[
                               int(y)] + "/" + str(count) + ".png")
                count+=1

        print(attack_name, "done")
