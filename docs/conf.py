# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the distribution-package root to sys.path so that the rsqsim_api
# package (and its subpackages) are importable during the Sphinx build.
# This is needed for type-hint resolution even though autoapi does its
# own static analysis without importing the code.
sys.path.insert(0, os.path.abspath("../src/rsqsim_api"))

# -- Project information -----------------------------------------------------

project = "rsqsim-python-tools"
copyright = "2022, Andy Howell and Camilla Penney"
author = "Andy Howell and Camilla Penney"
release = "0.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = False

# -- AutoAPI -----------------------------------------------------------------
# Point to the distribution-package root; autoapi discovers the rsqsim_api
# Python package (which contains __init__.py) inside it.
autoapi_dirs = ["../src/rsqsim_api"]
autoapi_type = "python"
autoapi_root = "autoapi"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

# Keep the generated .rst files so they can be inspected or customised.
autoapi_keep_files = True

# Let autoapi insert its own toctree entry after it has generated the
# files, avoiding the "nonexisting document 'autoapi/index'" warning
# that occurs when the entry is added manually before generation runs.
autoapi_add_toctree_entry = True

autodoc_typehints = "description"
