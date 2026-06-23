from setuptools import find_packages, setup
from typing import List

Hypen_dot='-e . '
def get_requirements(file_path:str)->List[str]:
    '''
    this function will require list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=[req.replace("\n", "") for req in file_obj.readlines()]
        if Hypen_dot in requirements:
            requirements.remove(Hypen_dot)
    return requirements

setup(
    name='MLproject',
    version='0.0.1',
    author='Tina',
    author_email='tinakoyal225@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)