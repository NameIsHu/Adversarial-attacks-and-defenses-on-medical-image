# Adversarial-attacks-and-defenses-on-medical-image
Code of implementation of adversarial attacks and defense on medical image 

## Requirements
Python >=3.8.15

pip install -r ./requirements.txt  or conda install --yes --file requirements.txt


## Datasets:
you can download datasets from :  

https://pan.baidu.com/s/1oJWMgtFtuGVijORpNMRJMA?pwd=7bhz   pwd: 7bhz 


## Usage

densenet.py resnet.py: create classfication model based on original datasets.

densenet_all.py resnet_all.py: create adversarial examples and test accuracy

densenet_all_save.py resnet_all_save.py: create and save adversarial examples

adversarial...  undercover...   : training adversarial defense model and test accuracy

## Publication

Our paper “A Multimodal Adversarial Database: Towards A Comprehensive Assessment of Adversarial Attacks and Defenses on Medical Images” has been accepted as a full paper by PSTDA@DSAA2023: The 5th edition of Special Session on Private, Secure, and Trust Data Analytics (PSTDA2023) @ the 10th IEEE International Conference on Data Science and Advanced Analytics (DSAA2023).
