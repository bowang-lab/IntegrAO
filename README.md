# IntegrAO
This is the official codebase for **Integrate Any Omics: Towards genome-wide data integration for patient stratification**.


## Requirements and installation
This package requires Python version 3.6 or greater. Assuming you have the correct version of Python, you can install this package by opening a command terminal and running the following:
```bash
git clone https://github.com/bowang-lab/IntegrAO.git
conda create -n integrAO python=3.9 -y
conda activate integrAO
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113  --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install -r requirement.txt
```

## Purpose
We introduce IntegrAO, an unsupervised framework integrating incomplete multi-omics and classifying new biological samples. 

![integrAO](https://github.com/bowang-lab/IntegrAO/blob/main/figures/integrAO_overview.png)

# :heavy_plus_sign: Tutorial

We offer the following tutorials for demonstration:

* [Integrate simulated butterfly datasets](https://github.com/bowang-lab/IntegrAO/blob/main/tutorials/simulated_butterfly.ipynb)


