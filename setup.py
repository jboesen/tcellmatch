from setuptools import setup, find_packages

author = 'theislab'
author_email='david.fischer@helmholtz-muenchen.de'
description="Predicting T-cell to epitope specificity."

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tcellmatch',
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    use_scm_version=True,
    python_requires='>=3.10',
    install_requires=[
        'anndata',
        'matplotlib',
        'numpy>=1.14.0',
        'pandas',
        'patsy',
        'scikit-learn',
        'scipy',
        'seaborn',
        'torch'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx-autodoc-typehints',
            'sphinx_rtd_theme',
            'jinja2',
            'docutils',
        ],
    },
    version="0.1.0",
)