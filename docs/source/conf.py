# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'parsernaam'
copyright = '2023, Rajashekar Chintalapati and Gaurav Sood'
author = 'Rajashekar Chintalapati and Gaurav Sood'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'myst_parser',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Type hints settings
typehints_defaults = 'comma'
typehints_use_signature = True
typehints_use_signature_return = True
autodoc_typehints_description_target = 'documented'
autodoc_type_aliases = {
    'DataFrame': 'pandas.DataFrame',
    'Tensor': 'torch.Tensor',
}

# MyST settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]

# Source file settings - support both markdown and rst
source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = []

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/appeler/parsernaam",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

html_title = f"{project} {release}"
html_logo = None
html_favicon = None

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOF"