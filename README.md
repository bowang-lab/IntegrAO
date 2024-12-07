# IntegrAO
This is the official codebase for **Integrate Any Omics: Towards genome-wide data integration for patient stratification**.

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://arxiv.org/abs/2401.07937) &nbsp;
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://scgpt.readthedocs.io/en/latest/) &nbsp;
[![PyPI version](https://badge.fury.io/py/scgpt.svg)](https://badge.fury.io/py/scgpt) &nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/bowang-lab/IntegrAO/blob/main/LICENSE)


## 🔨 Hardware requirements
`IntegrAO` package requires only a standard computer with enough RAM to support the in-memory operations.


## 🔨 Installation
This package requires Python version 3.6 or greater. If you want to utilize GPU computation, make sure you install the matching Pytorch and CUDA versions. 
1. Create a virtual environment:  `conda create -n integrAO python=3.10 -y` and `conda activate integrAO`
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.1.0: `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html): `pip install torch_geometric`
4. Install other requirements: `pip install -r requirement.txt`
5. Download code: `git clone https://github.com/bowang-lab/IntegrAO.git`


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
@article{ma2024integrate,
  title={Integrate Any Omics: Towards genome-wide data integration for patient stratification},
  author={Ma, Shihao and Zeng, Andy GX and Haibe-Kains, Benjamin and Goldenberg, Anna and Dick, John E and Wang, Bo},
  journal={arXiv preprint arXiv:2401.07937},
  year={2024}
}
```
