## Environment
### Install PyTorch
1. Python 3.10.13
2. Pytorch 2.5.1
3. torchvision: 0.20.1
4. torchaudio: 2.5.1
5. CUDA: 12.1
  
## Installation
### Install the remaining dependencies

pip install -r requirements.txt


## Train/Evaluate/Inference
### Example
```bash
python train.py --config configs/enh/cnn_aspp_test.yaml
python evaluate.py --config configs/enh/cnn_aspp_test.yaml
python inference.py --config configs/enh/cnn_aspp_test.yaml
