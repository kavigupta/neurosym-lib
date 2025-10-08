# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Import tomllib for Python 3.11+ or tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Read pyproject.toml
pyproject_path = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
with open(pyproject_path, "rb") as f:
    config = tomllib.load(f)

project_config = config["project"]

project = project_config["name"]
author = project_config["authors"][0]["name"]
copyright = "2024, " + author

release = version = project_config["version"]

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_static_path = ["../static"]

html_theme = "sphinx_rtd_theme"

html_style = "css/custom.css"

# -- Options for EPUB output
epub_show_urls = "footnote"

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]
