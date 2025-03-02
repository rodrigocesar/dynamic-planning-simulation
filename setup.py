from setuptools import setup, find_packages

setup(
    name="routing_simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "PyYAML",
    ],
) 