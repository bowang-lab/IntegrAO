# IntegrAO
This is the official codebase for **Integrate Any Omics: Towards genome-wide data integration for patient stratification**.


## 🔨 Installation
This package requires Python version 3.6 or greater. Assuming you have the correct version of Python, you can install this package by opening a command terminal and running the following:
```bash
git clone https://github.com/bowang-lab/IntegrAO.git
conda create -n integrAO python=3.9 -y
conda activate integrAO
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113  --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install -r requirement.txt
```

## 🧬 Introduction
High-throughput omics profiling advancements have greatly enhanced cancer patient stratification. However, incomplete data in multi-omics integration presents a significant challenge, as traditional methods like sample exclusion or imputation often compromise biological diversity and dependencies. Furthermore, the critical task of accurately classifying new patients with partial omics data into existing subtypes is commonly overlooked. We introduce IntegrAO, an unsupervised framework integrating incomplete multi-omics and classifying new biological samples. IntegrAO first combines partially overlapping patient graphs from diverse omics sources and utilizes graph neural networks to produce unified patient embeddings.

An overview of IntegrAO can be seen below.

![integrAO](https://github.com/bowang-lab/IntegrAO/blob/main/figures/integrAO_overview.png)

## 📖 Tutorial

We offer the following tutorials for demonstration:

* [Integrate simulated butterfly datasets](https://github.com/bowang-lab/IntegrAO/blob/main/tutorials/simulated_butterfly.ipynb)
* [Integrate simulated cancer omics datasets](https://github.com/bowang-lab/IntegrAO/blob/main/tutorials/simulated_cancer_omics.ipynb)
* [Classify new samples with incomplete omics datasets](https://github.com/bowang-lab/IntegrAO/blob/main/tutorials/cancer_omics_classification.ipynb)

## Citing IntegrAO
```bash

```
