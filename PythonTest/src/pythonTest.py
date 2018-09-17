#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Function:
【记录】使用Python的IDE：Eclipse+PyDev
 
http://www.crifan.com/try_with_python_ide_eclipse_pydev
 
Author:     Crifan Li
Version:    2012-12-29
Contact:    admin at crifan dot com
"""

s="baidu ZHIDAO"
print(s.upper()+"\n")
print(s.lower()+"\n")
print(s.swapcase()+"\n")


print("hello")
#import platform;

print ("Hello, Python!");
 
#print ("Current platform.uname() in Ecplise+PyDev=" ),platform.uname();

temp = input("please enter a number:\n");

number=int(temp);

if number == 8:
    print("ok!");
else:
    print("no");
print("--------------------------------------------------");

import sys
print("Python version:{}".format(sys.version))

import pandas as pd
print("pandas version:{}".format(pd.__version__))

import matplotlib
print("matplotlib version:{}".format(matplotlib.__version__))

import numpy as np
print ("Numpy version:{}".format(np.__version__))

import scipy as sp
print("Scipy version:{}".format(sp.__version__))

import IPython
print("IPython version:{}".format(IPython.__version__))

import sklearn
print("scikit-learn version:{}".format(sklearn.__version__))


