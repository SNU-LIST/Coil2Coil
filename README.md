# Coil2Coil
* The code is for denoising of MR images using deep learning referred to as Coil2Coil (C2C). C2C 
generated paired noise-corrupted images from phased-array coil data to train a deep neural network, and the paried images were modified to satisfy the conditions of Noise2Noise (N2N; Lehtinen, et al), enabling network training using N2N algorithm. 
* last update : 2022. 08. 07

# Reference
* will be updated

# Overview

![figure 1](/figure.png)

# Requirements 
* Python 3.7

* pytorch=1.9.0

* NVIDIA GPU 


# Usage
### Installation

First, download the codes using the command below. 

```python
git clone https://github.com/SNU-LIST/Coil2Coil.git
```

Then, go inside 'Coil2Coil' file, and install the ssmd packages

Following command will be install a few dependent libraries such as Pytorch, Numpy, and etc...
```python
pip install -e ssmd
```
There might be 'ssmd' library on your environment (pip list)


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

Run the train.py with following codes

Arguments are provided below:
```python
python train.py -dir {directory to save logs} 
                -a {algorithm, 'c2c' or 'n2c'} 
                -t {path for training dataset} 
                -v {path for validation dataset} 
                -noise {amount of noise (inf or gauss(float))}
```
# LIcense
We provide software for academic research purpose only and NOT for commercial or clinical use.

For commercial use of our software, contact us (snu.list.software@gmail.com) for licensing via Seoul National University.

Please email to “snu.list.software@gmail.com” with the following information.

Name:

Affiliation:

Software:

When sending an email, an academic e-mail address (e.g. .edu, .ac.) is required.

# Contact 
Juhyug Park, M.S-Ph.D candidate, Seoul National University

jack0878@snu.ac.kr

http://list.snu.ac.kr
