# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'IntegrAO'
copyright = '2024, Wanglab, Shihao Ma'
author = 'Shihao Ma'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'nbsphinx',
    'nbsphinx_link',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme' #To use this theme, make sure 'sphinx' and 'sphinx_rtd_theme' are in requirements.txt AND sphinx_rtd_theme is listed in extensions above (https://sphinx-rtd-theme.readthedocs.io/en/stable/installing.html)

# -- Options for EPUB output
epub_show_urls = 'footnote'
