"""
Copyright 2021 Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '2021.11.16.0'

requirements = [
    'torch'
]

setup(
    # Metadata
    name='haar_pytorch',
    version=VERSION,
    author='Sergei Belousov aka BeS',
    author_email='sergei.o.belousov@gmail.com',
    url='https://github.com/bes-dev/haar_pytorch',
    description='Pytorch implementation of the forward and inverse discrete wavelet transform using Haar Wavelets.',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache 2.0',

    # Package info
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],

    tests_require=['transform2d'],
)
