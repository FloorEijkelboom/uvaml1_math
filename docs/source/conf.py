# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Math (MSc AI, UvA)'
copyright = '2022, Floor Eijkelboom'
author = 'Floor Eijkelboom'

release = '1.0'
version = '1.0.0'

# -- General configuration

extensions = [
	'nbsphinx',
	'sphinx.ext.mathjax',
   'sphinx.ext.autosectionlabel',
]

# Make sure the target is unique
autosectionlabel_prefix_document = True


mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "{number}"



intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    app.add_css_file('style.css')
