# smartcontract
Ethereum smart contract similarity measurement at the basic block level based on triplet netwok
The corresponding Ethereum smart contract source code and data set are in：https://drive.google.com/file/d/1BniufNIGWk1wR0yKpYNDeNlhJcqlLlbA/view?usp=sharing
Requirements and Dependencies
Ubuntu (We test with Ubuntu = 18.04 LTS)
Python (We test with Python = 3.7.4)
CUDA & cuDNN (We test with CUDA = 10.1 and cuDNN = 7.6.5)
PyTorch （We test with PyTorch = 1.0.0）
NVIDIA GPU(s) (We use 4 RTX 2080Ti)
Modify the data set path in data_manager.py 
python  data_manager.py
python  pretrain.py
python train.py
