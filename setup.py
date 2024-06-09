from setuptools import find_packages,setup
from typing import List

# code for not to input or skip or ignore the -e . in requirement files as it appear in the list of packages which are going to install
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this funtion will return the list of requiremnets
    '''
    requirements=[]
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

setup(
    name='mlproject',
    version='0.0.1',
    author='sanjaychilveri',
    author_email='chilverisanjay@gmail.com',
    packages= find_packages(),
    # install_requires=['pandas', 'numpy', 'seaborn'] --> this is not feaasible for many packages 
    # to avoid this problem we create a function
    install_requirements=get_requirements('requirements.txt')

)