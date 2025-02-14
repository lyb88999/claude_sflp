from setuptools import setup, find_packages

setup(
    name="satellite_fl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'skyfield',
        'astropy',
        'matplotlib',
        'pyyaml',
        'networkx',
        'scipy',
        'torch',
        'scikit-learn'
    ],
)
