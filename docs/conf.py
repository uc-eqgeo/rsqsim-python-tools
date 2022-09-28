# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/rsqsim_api/rsqsim_api/"))

project = 'rsqsim-python-tools'
copyright = '2022, Andy Howell and Camilla Penney'
author = 'Andy Howell and Camilla Penney'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
autoapi_dirs = ["../src/rsqsim_api/rsqsim_api/"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autodoc_typehints = "description"