# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:20:45 2019

@author: luyao.li
"""
from  setuptools import  setup
with  open('README.md','r') as f:
    LONG_DESCRIPTION=f.read()
setup(
      name='light_famd',
      version='0.0.1',
      keywords=['famd','factor analysis'],
      description='Light Factor Analysis of Mixed Data',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      packages=['light_famd'],
      install_requires=["scikit-learn","scipy","pandas","numpy"],
      py_modules=['numbers','math'],
      url='https://github.com/Cauchemare/Truncated_FAMD',
      author='telescopes',
      author_email='luyaoli88@gmail.com',
      classifiers=[
              'Development Status :: 3 - Alpha',
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: BSD License ",
              "Operating System :: OS Independent",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Software Development :: Libraries :: Python Modules",
              "Intended Audience :: Science/Research"])


