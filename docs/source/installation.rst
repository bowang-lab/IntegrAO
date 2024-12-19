Installation
------------
IntegrAO works with Python >= 3.7. Please make sure you have the correct version of Python pre-installation.

1. Create a virtual environment
::
   conda create -n integrAO python=3.7 -y
   conda activate integrAO

2. Install `Pytorch <https://pytorch.org/get-started/locally/>`_ 2.1.0
::
   pip install torch torchvision torchaudio

3. IntegrAO is available on PyPI. To install IntegrAO, run the following command
::
   pip install integrao

For developing, clone this repo with following commands::

   git clone https://github.com/bowang-lab/IntegrAO.git
   cd IntegrAO
   pip install -r requirement.txt
