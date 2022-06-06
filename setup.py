import os
from setuptools import setup
from hftrim import __version__

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

readme_fname = os.path.join(THIS_DIR, 'README.md')
with open(readme_fname, 'r') as f:
    long_description = f.read()

setup(
    name='hf-trim',
    version=__version__,    
    description='A tool to reduce the size of Hugging Face models via vocabulary trimming.',
    url='https://github.com/IamAdiSri/hf-trim',
    download_url='https://github.com/IamAdiSri/hf-trim/archive/v2.3.1-beta.tar.gz',
    author='Aditya Srivastava',
    author_email='adi.srivastava@hotmail.com',
    license='MPL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=[
        'hftrim'
    ],
    install_requires=[
        'numpy>=1.22.3',
        'protobuf>=3.19.4',
        'tokenizers>=0.11.6',
        'torch>=1.11.0',
        'tqdm>=4.63.1',
        'transformers>=4.17.0',
        'sentencepiece>=0.1.96'
    ],
    python_requires='>=3.8',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)