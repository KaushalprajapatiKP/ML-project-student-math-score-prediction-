"""Setup.py file is for exporting this whole project as a packagae file and can upload on pypi and anybody can download it from there
and use it as package. Basically building project as package itselt."""

from setuptools import find_packages,setup
from typing import List

Hyphen_E_Dot = "-e ."
def get_requirements(file_path: str) -> List[str]:
    #This function will return a list of requirements
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    if Hyphen_E_Dot in requirements:
        requirements.remove(Hyphen_E_Dot)
    
    return requirements

setup(
    name="Machine learning Project",
    version="0.0.1",
    author="Kaushal Prajapati",
    author_email="kaushalprajapati5296@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt"),
)