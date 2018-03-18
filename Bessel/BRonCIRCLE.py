# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 22:07:09 2015

@author: raghavsinghal
"""

from numba import jit,float64,int64
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def BRonC(N,delta,T):
    z=np.arange(N)/float(N)
    e=0.1
    m=4
    x=(1. + e*np.sin(2*pi*m*z))*np.cos(2*pi*z)
    y=(1. + e*np.sin(2*pi*m*z))*np.sin(2*pi*z)
    
    
    x,y=RKF(x,y,z,T,delta)
    plt.figure(1)
    plt.scatter(x,y)
    
    
#@jit(float64(float64[:],float64[:],float64[:],int64,float64))                 
def RKF(x,y,z,T,delta):
    etol=1e-2
    dt=0.05
    
    y=np.vstack((x,y))    
        
    u=y
    N=np.array([len(x)])
    time=np.zeros(1)

    K1=np.zeros((2,1))
    K2=np.zeros((2,1))
    K3=np.zeros((2,1))
    K4=np.zeros((2,1))
    K5=np.zeros((2,1))
    K6=np.zeros((2,1))
    n=0.0
    
    
    a2=0.25

    b2=3./32.
    b3=9./32.

    c2=1932./2197.
    c3=-7200./2197.
    c4=7296./2197.

    d2=439./216.
    d3=-8.
    d4=3680./513.
    d5=-845./4104.

    e2=-8./27.
    e3=2
    e4=-3544./2565.
    e5=1859./4104.
    e6=-11./40.
    
    r1=1./360.
    r3=-128./4275.
    r4=-2197./75420.
    r5=1./50.
    r6=2./55.
    
    a11=16./135.
    a13=6656./12825.
    a14=28561./56430.
    a15=-9./50.
    a16=2./55.
    
    for t in time:
        while t<=T:# and n*dt<=T:
              K1=dt*f(t,y,delta)

              K2=dt*f(t + 0.25*dt, y + a2*K1,delta)

              K3=dt*f(t + (3./8.)*dt, y + b2*K1 + b3*K2,delta)

              K4=dt*f(t + (12./13.)*dt, y + c2*K1 + c3*K2 + c4*K3,delta)
              
              K5=dt*f(t + dt, y + d2*K1 + d3*K2 + d4*K3 + d5*K4,delta)
              
              K6=dt*f(t + 0.5*dt, y + e2*K1 + e3*K2 + e4*K3 + e5*K4 + e6*K5,delta)
              
              error=np.max(abs(r1*K1 + r3*K3 + r4*K4 + r5*K5 + r6*K6)/dt)
              
              if (error)<etol or dt<1e-3:
                 y=y + a11*K1 + a13*K3 + a14*K4 + a15*K5 + a16*K6
                                  
                 u=np.hstack((u,y))
                 time=np.append(time,t)
                 t=t+dt
                 n+=1
                 N=np.append(N,len(y[0,:]))
                 if n%1==0:  
                    print (T-t),n
                 
              else:
                 dt=0.84*dt*((etol/(error))**(0.25))
                 print (dt)

    np.savetxt('XdataCircle.txt',u[0,:])
    np.savetxt('YdataCircle.txt',u[1,:])
    np.savetxt('NdataCircle.txt',N)
    np.savetxt('NumberOfIterationCircle.txt',np.array([n-1]))

    return y[0,:],y[1,:]                 

    
@jit(float64(int64,float64[:,:],float64))             
def f(t,X,delta):
    x=X[0,:]
    y=X[1,:]
    
    N=len(x)
    n=float(N)
    dx,dy=np.ones(N),np.ones(N)
    for j in range(N):       
        for k in range(N):   
            num1=-(y[j]-y[k])
            num2=(x[j]-x[k])
            denom=((y[j]-y[k])**2 + (x[j]-x[k])**2 + delta**2)
            dx[j]+=num1/denom
            dy[j]+=num2/denom
    
    return np.vstack((0.5*dx/n,0.5*dy/n))
