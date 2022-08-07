# CoiltoCoil
* The code is for denoising of MR images using deep learning (C2C), C2C 
generated paired noise-corrupted images from phased-array coil data to train a deep neural network. Then the paried images were modified to satisfy the conditions of Noise2Noise (Lehtinen, et al), enabling network training using N2N algorithm. 
* last update : 2022. 08. 07

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
