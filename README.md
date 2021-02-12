# smartcontract

The corresponding Ethereum smart contract source code and data set are in：https://drive.google.com/file/d/1BniufNIGWk1wR0yKpYNDeNlhJcqlLlbA/view?usp=sharing
### Table of Contents
1. [Introduction](#introduction)
2. [Requirements and Dependencies](#requirements-and-dependencies)
3. [Data Preprocessing](#data_preprocessing)
4. [Pretrain](#pretrain)
5. [Train](#prain) 

### Introduction
Ethereum smart contract similarity measurement at the basic block level based on triplet netwok.

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 18.04 LTS)
- Python (We test with Python = 3.7.4)
- CUDA & cuDNN (We test with CUDA = 10.1 and cuDNN = 7.6.5)
- PyTorch （We test with PyTorch = 1.6.0）
- NVIDIA GPU(s) (We use 4 RTX 2080Ti)



### Data Preprocessing



```
$ python data_manager.py
```

### Pretrain

```
$ python pretrain.py
```

### train

```
$ python train.py
```

### Contact
[Di Zhu](mailto:245563617@qq.com)

