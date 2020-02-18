name = 'ulat'

import os
from setuptools import setup, Extension, Command
import sys

setup(name = name)

pjoin = os.path.join
here = os.path.abspath(os.path.dirname(__file__))

packages = []
for d, _, _ in os.walk(pjoin(here, name)):
    if os.path.exists(pjoin(d, '__init__.py')):
        packages.append(d[len(here)+1:].replace(os.path.sep, '.'))

version_ns = {}
with open(pjoin(here, name, '_version.py')) as f:
    exec(f.read(), {}, version_ns)

setup_args = dict(
    name            = name,
    version         = version_ns['__version__'],
    packages        = packages,
    description     = "",
    long_description= "",
    author          = 'Clément Ranc',
    author_email    = 'clement.ranc@protonmail.com',
    url             = '',
    license         = 'MIT',
    platforms       = "Linux, Mac OS X, Windows",
    keywords        = ['Astronomy', 'Microlensing', 'Science'],
    classifiers     = [
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
                       ],
)

if 'develop' in sys.argv or any(a.startswith('bdist') for a in sys.argv):
    import setuptools

setuptools_args = {}

install_requires = setuptools_args['install_requires'] = [
                                                          'bokeh>=0.12.4',
                                                          'sphinx>=2.4',
                                                          'sphinx_press_theme'
]

extras_require = setuptools_args['extras_require'] = {
}

if 'setuptools' in sys.modules:
    setup_args.update(setuptools_args)

if __name__ == '__main__':
    setup(**setup_args)
