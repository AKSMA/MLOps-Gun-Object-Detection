from setuptools import setup, find_packages 

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ML_OPs pROJECT 4",
    version = "0.1", 
    author = "Akshat" ,
    packages = find_packages(),
    install_requires = requirements, 
)