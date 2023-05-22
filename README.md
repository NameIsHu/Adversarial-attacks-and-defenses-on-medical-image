# Adversarial-attacks-and-defenses-on-medical-image
Code of implementation of adversarial attacks and defense on medical image 

## Requirements
Python >=3.8.15

pip install -r ./requirements.txt  or conda install --yes --file requirements.txt


## Datasets:
you can download datasets from :  链接: https://pan.baidu.com/s/1oJWMgtFtuGVijORpNMRJMA?pwd=7bhz 提取码: 7bhz 复制这段内容后打开百度网盘手机App，操作更方便哦


## Usage

densenet.py resnet.py: create classfication model based on original datasets.

densenet_all.py resnet_all.py: create adversarial examples and test accuracy

densenet_all_save.py resnet_all_save.py: create and save adversarial examples

adversarial...  undercover...   : training adversarial defense model and test accuracy
