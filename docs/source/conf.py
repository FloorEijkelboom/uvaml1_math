# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Math notebooks'
copyright = '2021, Floor Eijkelboom'
author = 'Floor Eijkelboom'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
	'nbsphinx',
	'sphinx.ext.mathjax',
]


mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
mathjax3_config = {
    "TeX": {
        #"packages": {'[+]': ['bm']},
        "Macros": {
            "paran": ['{\\left( #1 \\right) }', 1],
            "braket": ['{\\left[ #1 \\right] }', 1],
            "bold": ['{\\bf #1}', 1],
            "R": '{\\mathbb R}',
            "N": '{\\mathbb N}',
            "F": '{\\mathbb F}',
            "Z": '{\\mathbb Z}',
            "inv": '{^{-1}}',
            "matrix": ['{\\begin{bmatrix} #1 \\end{bmatrix}}', 1],
            # bold letters for vectors
            "va": '{\\bf a}',
            "vb": '{\\bf b}',
            "vc": '{\\bf c}',
            "vd": '{\\bf d}',
            "ve": '{\\bf e}',
            "vf": '{\\bf f}',
            "vg": '{\\bf g}',
            "vh": '{\\bf h}',
            "vi": '{\\bf i}',
            "vj": '{\\bf j}',
            "vk": '{\\bf k}',
            "vl": '{\\bf l}',
            "vm": '{\\bf m}',
            "vn": '{\\bf n}',
            "vo": '{\\bf o}',
            "vp": '{\\bf p}',
            "vq": '{\\bf q}',
            "vr": '{\\bf r}',
            "vs": '{\\bf s}',
            "vt": '{\\bf t}',
            "vu": '{\\bf u}',
            "vv": '{\\bf v}',
            "vw": '{\\bf w}',
            "vx": '{\\bf x}',
            "vy": '{\\bf y}',
            "vz": '{\\bf z}',
        },
    }
}



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
