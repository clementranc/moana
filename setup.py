import os
from setuptools import setup

setup(name='moana-pypi',
      version='0.2.1',
      packages=['moana',
                'moana.corner',
                'moana.dbc',
                'moana.stylelib'],
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
