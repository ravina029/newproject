from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e'

def get_requirements(file_path: str) -> List[str]:
    """This function will return the list of requirements"""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # requirements = [req.replace('\n'," ")for req in requirements]
        requirements = [req.strip() for req in requirements]
        requirements = [req for req in requirements if req and not req.startswith(HYPHEN_E_DOT)]

    return requirements

setup(
    name='mlproject',
    version='0.01',
    author='Ravina',
    author_email='vermaravina029@gmail.com',
    packages=find_packages(),
    # install_requires=['pandas', 'numpy', 'seaborn'],
    install_requires=get_requirements('requirements.txt')
)
