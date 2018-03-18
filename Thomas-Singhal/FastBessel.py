# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:54:20 2015

@author: raghavsinghal
"""

import numpy  as np
from math import log,sinh,atan2,cos
from scipy.special import k1

r=1.2

def FastBessel(r):
    a=atan2(4,log(sinh(r)))
    b=0.534 - 0.6*cos(a) -0.068*cos(2*a) + 0.125*cos(3*a) + 0.032*cos(4*a)-0.032*cos(5*a)
    return b/r
    
print k1(r)-FastBessel(r) 