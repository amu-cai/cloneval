from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cloneval",
    version="0.1.0",
    author="Center for Artificial Intelligence",
    author_email="iwona.christop@amu.edu.pl",
    description="ClonEval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amu-cai/cloneval",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "librosa",
        "torch",
    ],
)