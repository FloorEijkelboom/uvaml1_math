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
            # bold capital letters for matrices
            "mA": '{\\bf A}',
            "mB": '{\\bf B}',
            "mC": '{\\bf C}',
            "mD": '{\\bf D}',
            "mE": '{\\bf E}',
            "mF": '{\\bf F}',
            "mG": '{\\bf G}',
            "mH": '{\\bf H}',
            "mI": '{\\bf I}',
            "mJ": '{\\bf J}',
            "mK": '{\\bf K}',
            "mL": '{\\bf L}',
            "mM": '{\\bf M}',
            "mN": '{\\bf N}',
            "mO": '{\\bf O}',
            "mP": '{\\bf P}',
            "mQ": '{\\bf Q}',
            "mR": '{\\bf R}',
            "mS": '{\\bf S}',
            "mT": '{\\bf T}',
            "mU": '{\\bf U}',
            "mV": '{\\bf V}',
            "mW": '{\\bf W}',
            "mX": '{\\bf X}',
            "mY": '{\\bf Y}',
            "mZ": '{\\bf Z}',
        },
    }
}

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