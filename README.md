# Variational Attention
![](https://img.shields.io/badge/python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.10.0-orange.svg)

This is the augmented codebase for the following paper, implemented in tensorflow:

Hareesh Bahuleyan*, Lili Mou*, Olga Vechtomova, and Pascal Poupart. **Variational Attention for Sequence-to-Sequence Models.** COLING 2018. https://arxiv.org/pdf/1712.08207.pdf

## Overview
...

## Datasets
...


## Requirements
- tensorflow-gpu==1.10.0
- Keras==2.1.6
- numpy==1.14.3
- pandas==0.22.0
- gensim==3.4.0
- nltk==3.2.5
- tqdm==4.23.1

## Instructions
1. Generate word2vec, required for initializing word embeddings, specifying the dataset:
```
python w2v_generator.py 
```
2. Train the desired model, set configurations in the `configuration.py` file. For example,
```
cd ved_varAttn
vim configuration.py # Make necessary edits
python train.py
``` 
- ... 
