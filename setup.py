from os.path import abspath, dirname, join
from setuptools import setup, find_packages

here = abspath(dirname(__file__))

setup(
    name='wgan',
    version='0.3',
    description='Optimized WGAN for simulating economic datasets',
    author='Jonas Metzger, Evan Munro (optimized)',
    author_email='munro@stanford.edu',
    long_description='Optimized WGAN package with performance improvements for training speed.',
    long_description_content_type='text/markdown',
    url='https://github.com/gsbDBI/ds-wgan',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=1.1.0",
        "pandas",
        "matplotlib"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    license='MIT',
)
