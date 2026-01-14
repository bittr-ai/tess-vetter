# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add the source directory to sys.path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Try to read metadata from pyproject.toml
try:
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
    project_meta = pyproject.get("project", {})
    project = project_meta.get("name", "bittr-tess-vetter")
    release = project_meta.get("version", "0.0.1")
except Exception:
    project = "bittr-tess-vetter"
    release = "0.0.1"

copyright = "2024, bittr-tess-vetter contributors"
author = "bittr-tess-vetter contributors"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Autosummary settings
autosummary_generate = True
# We list the public surface explicitly in `docs/api.rst`; avoid pulling in
# imported members automatically (it can create duplicate/unstable pages).
autosummary_imported_members = False

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
# Render attribute documentation as `:ivar:` fields rather than creating
# separate domain objects. This avoids duplicate-object warnings for dataclass/
# pydantic model fields when combined with autosummary.
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    # Keep autosummary pages concise; our public API listing lives in api.rst.
    # Documenting members on large schema classes generates many duplicate
    # object descriptions (Class.field) and noisy docutils warnings.
    "members": False,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
root_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4d9fff",
        "color-brand-content": "#4d9fff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Additional HTML settings
html_title = f"{project} v{release}"
html_short_title = project

# -- Options for autodoc -----------------------------------------------------

# Suppress warnings about missing references for optional dependencies
nitpicky = False

# Mock imports for optional dependencies that may not be installed during docs build
autodoc_mock_imports = [
    "mlx",
    "mlx.core",
    "wotan",
    "ldtk",
    "lightkurve",
    "triceratops",
    "pytransit",
    "mechanicalsoup",
    "seaborn",
    "pyrr",
    "celerite",
    "corner",
]
