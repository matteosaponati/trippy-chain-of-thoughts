from setuptools import find_packages, setup

setup(
    name = 'trippy-chain-of-thoughts',
    version = '0.1.0',
    description = 'A package for running trial experiments and making an LLM trip like hell',
    packages = find_packages(),
    python_requires = '>=3.11',
)