.. SHAP documentation master file,
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IntegrAO's documentation
===================================


**IntegrAO** (**Integr**\ate **A**\ny **O**\mics) is an unsupervised, GNN-based framework for integrating incomplete multi-omics data. (see
`paper <https://arxiv.org/abs/2401.07937>`_ for details and citations).

.. image:: https://img.shields.io/badge/preprint-available-brightgreen.svg?style=flat
    :target: https://arxiv.org/abs/2401.07937
    :alt: Preprint link

.. image:: https://badge.fury.io/py/integrao.svg
      :target: https://badge.fury.io/py/integrao
      :alt: PyPI version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
      :target: https://github.com/bowang-lab/IntegrAO/blob/main/LICENSE
      :alt: License


Introduction
----------
High-throughput omics profiling advancements have greatly enhanced cancer patient stratification. However, incomplete data in multi-omics integration presents a significant challenge, as traditional methods like sample exclusion or imputation often compromise biological diversity and dependencies. Furthermore, the critical task of accurately classifying new patients with partial omics data into existing subtypes is commonly overlooked. We introduce IntegrAO, an unsupervised framework integrating incomplete multi-omics and classifying new biological samples. IntegrAO first combines partially overlapping patient graphs from diverse omics sources and utilizes graph neural networks to produce unified patient embeddings.

Overview
--------
.. image:: https://github.com/bowang-lab/IntegrAO/raw/main/figures/integrAO_overview.png


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Installation <installation>

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial_butterfly
   tutorial_cancer
   tutorial_classify

.. toctree::
   :maxdepth: 2
   :caption: API

   integrao

.. toctree::
   :maxdepth: 2
   :caption: References:

   faq
   references

