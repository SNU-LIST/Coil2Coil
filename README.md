# CoiltoCoil
* The code is for denoising of MR images using deep learning (C2C)
* last update : 2022. 07. 04

# Overview

![figure 1](/fig1.png)

# Requirements 
* Python 3.7

* pytorch=1.9.0

* NVIDIA GPU 


# Usage
### Installation

First, download the codes. 

```python
git clone https://github.com/SNU-LIST/CoiltoCoil.git
```

Then, go inside 'CoiltoCoil' file, and install the ssmd packages

Following command will be install a few libraries such as Pytorch, Numpy, and etc...
```python
pip install -e ssmd
```
There might be 'ssmd' library (pip list)

### Inference

Run the eval.py with following codes
Arguments are provided below:
```python
python eval.py -dir {directory to save logs} 
               -m {model weight path} 
               -d {dataset path for inference} 
               -noise {amount of noise (inf for just inference)}
```
### Traininig



