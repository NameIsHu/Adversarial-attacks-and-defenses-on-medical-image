from ast import Tuple
from re import T
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
class1=[0,1,2,3,4]
if __name__ == '__main__':
    transforming_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=5)
    model.load_state_dict(torch.load('./model/resnet2.pth', map_location="cpu"))
    device = torch.device('cuda')
    cpu_device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    print("use resnet model for Diabetic Retinopathy")
    attacks = {}
    attacks["PGD"] = 1
    attacks["FGSM"] = 1
    attacks["CW"] = 1
    attacks["BIM"] = 1
    attacks["DeepFool"] =1
    attacks["OnePixel"] = 1

    for attack_name, attack in attacks.items():
        test_path = ('./Adversarial Image/densenet/'+attack_name+'/testing/')
        test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transforming_img),
        batch_size=512, shuffle=True
    )
        y_o_tr=[]
        y_p_tr=[]
        y_o_f1=[]
        y_p_f1=[]
        y_o_auc=[]
        y_p_auc=[]
        total = 0
        correct = 0
        for i, (atk_images, atk_labels) in enumerate(test_loader):
            atk_images = atk_images.to(device)
            atk_labels = atk_labels.to(device)
            adv_op = model(atk_images)
            _, p_adv = torch.max(adv_op.data, dim=1)
            correct += int((p_adv == atk_labels).sum())
            total += len(adv_op)
            
            a=nn.Softmax(dim=1)(adv_op).data
            y_o_auc.extend(atk_labels.tolist())
            y_p_auc.extend(a.tolist())        

            y_o_f1.extend(atk_labels.tolist())
            y_p_f1.extend(p_adv.tolist())

        for i in range(len(y_o_f1)):
            y_o_f1[i]=[y_o_f1[i]]
            y_p_f1[i]=[y_p_f1[i]]

        y_o_f1=MultiLabelBinarizer(classes=class1).fit_transform(y_o_f1)
        y_p_f1=MultiLabelBinarizer(classes=class1).fit_transform(y_p_f1)
        print("test data num:", total, "Test auc after ", attack_name, " attack:", correct/total,flush=True)
        print("test data num:", total, "Test f1 after ", attack_name, " attack:", f1_score(y_o_f1,y_p_f1,average='weighted'),flush=True)
        print("test data num:", total, "Test auc after ", attack_name, " attack:", roc_auc_score(y_o_auc,y_p_auc,average='weighted',multi_class="ovr"),flush=True)

