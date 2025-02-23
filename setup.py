from setuptools import setup, find_packages

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.18",
        "python-json-logger"
    ],
)