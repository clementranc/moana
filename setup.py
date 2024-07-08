import os
from setuptools import setup

name='moana-pypi'
packages = []
pjoin = os.path.join
here = os.path.abspath(os.path.dirname(__file__))
for d, _, _ in os.walk(pjoin(here, name)):
    if os.path.exists(pjoin(d, '__init__.py')):
        packages.append(d[len(here) + 1:].replace(os.path.sep, '.'))

setup(name=name,
      version='0.2',
      packages=packages,
      description="",
      long_description="",
      author='Clément Ranc, Stela Ishitani Silva, MOAna authors',
      author_email='ranc@iap.fr',
      maintainer_email= 'stela.ishitanisilva@nasa.gov',
      url='',
      license='MIT',
      platforms="Linux, Mac OS X, Windows",
      keywords=['Astronomy', 'Microlensing', 'Science'],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: System Administrators',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.11',
      ],
      install_requires=[
          'wheel==0.43.0',
          'scipy==1.13.1',
          'sphinx==7.3.7',
          'sphinx_press_theme==0.9.1'])
