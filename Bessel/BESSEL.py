# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:44:05 2015

@author: raghavsinghal
"""
from numba import jit,float64,int64
import numpy as np
from scipy.special import k1
import matplotlib.pyplot as plt
from math import pi
from math import sqrt
import scipy.interpolate as interpolate



def Bessel(N,delta,Kernel,m,T,alpha):
    e=0.1/m
    
    z=np.arange(N)/float(N)

    x=(1. + e*np.sin(2*pi*m*z))*np.cos(2*pi*z)
    y=(1. + e*np.sin(2*pi*m*z))*np.sin(2*pi*z)
    
    x,y,u=RKF(x,y,z,T,delta,Kernel,alpha)
    
    return x,y,u


    
@jit(float64(float64[:],float64[:],float64[:],int64,float64,float64,float64))                 
def RKF(x,y,z,T,delta,Kernel,alpha):
    etol=1e-3
    
    dt=0.1
    #dt=delta*0.5
    
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
        while t<=T and n*dt<=T:
              K1=dt*f(t,y,delta,Kernel,alpha)

              K2=dt*f(t + 0.25*dt, y + a2*K1,delta,Kernel,alpha)

              K3=dt*f(t + (3./8.)*dt, y + b2*K1 + b3*K2,delta,Kernel,alpha)

              K4=dt*f(t + (12./13.)*dt, y + c2*K1 + c3*K2 + c4*K3,delta,Kernel,alpha)
              
              K5=dt*f(t + dt, y + d2*K1 + d3*K2 + d4*K3 + d5*K4,delta,Kernel,alpha)
              
              K6=dt*f(t + 0.5*dt, y + e2*K1 + e3*K2 + e4*K3 + e5*K4 + e6*K5,delta,Kernel,alpha)
              
              error=np.max(abs(r1*K1 + r3*K3 + r4*K4 + r5*K5 + r6*K6)/dt)
              
              if (error)<etol or dt<1e-3:
                  
                 y=y + a11*K1 + a13*K3 + a14*K4 + a15*K5 + a16*K6
                        
                 y,z=meshrefinement(y,z)
                 
                 u=np.hstack((u,y))
                 time=np.append(time,t)
                 N=np.append(N,len(y[0,:]))

                 if n%20==0:  
                    print (T-t),len(y[0,:])
                 
                 t=t+dt
                 n+=1              
                 
                 
              else:
                 dt=0.84*dt*((etol/(error))**(0.25))
                 print (dt)

    np.savetxt('XdataBessel.txt',u[0,:])
    np.savetxt('YdataBessel.txt',u[1,:])
    np.savetxt('NdataBessel.txt',N)
    np.savetxt('NumberOfIterationBessel.txt',np.array([n-1]))
    print 'saved'
    
    return y[0,:],y[1,:],u             
    

@jit(float64(int64,float64[:,:],float64,int64,float64))             
def f(t,X,delta,Kernel,alpha):
    x=X[0,:]
    y=X[1,:]
    
    
    N=len(x)
    n=float(N)
    dx,dy=np.ones(N),np.ones(N)
    if Kernel==1:
        for j in range(N):       
            for k in range(j,N):   
                num1=-(y[j]-y[k])
                num2=(x[j]-x[k])
                denom=((y[j]-y[k])**2 + (x[j]-x[k])**2 + delta**2)
                dx[j]+=num1/denom
                dy[j]+=num2/denom
                
                dx[k]+=-num1/denom
                dy[k]+=-num2/denom
                
    elif Kernel==0:  
        for j in range(N):       
            for k in range(j,N):   
                num1=-alpha*(y[j]-y[k])*k1(alpha*sqrt((y[j]-y[k])**2 + (x[j]-x[k])**2 + delta**2))
                num2=alpha*(x[j]-x[k])*k1(alpha*sqrt((y[j]-y[k])**2 + (x[j]-x[k])**2 + delta**2))
                denom=sqrt((y[j]-y[k])**2 + (x[j]-x[k])**2 + delta**2)
                dx[j]+=num1/denom
                dy[j]+=num2/denom
                
                dx[k]+=-num1/denom
                dy[k]+=-num2/denom
    
    return np.vstack((0.5*dx/n,0.5*dy/n))

@jit(float64(float64[:,:],float64[:]))
def meshrefinement(u,z):
    x=u[0,:]
    y=u[1,:]
    t=0    
    etolmin=0.005
    etolmax=0.02
    
    while t<len(x)-1:
        if sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)>=etolmax:
            xtck=interpolate.splrep(z,x,k=3)
            ytck=interpolate.splrep(z,y,k=3)
            
            znew=0.5*(z[t] + z[t+1])
            
            xnew=interpolate.splev(znew,xtck)
            ynew=interpolate.splev(znew,ytck)
            
            x=np.insert(x,t+1,xnew)
            y=np.insert(y,t+1,ynew)
            z=np.insert(z,t+1,znew)
            
        elif sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)<=etolmin:
            
            x=np.delete(x,t+1)
            y=np.delete(y,t+1)
            z=np.delete(z,t+1)
            
        t+=1
    
    xtck=interpolate.splrep(z,x,k=3)
    ytck=interpolate.splrep(z,y,k=3)
    z=np.arange(len(x))/float(len(x))
    
    x=interpolate.splev(z,xtck)
    y=interpolate.splev(z,ytck)


    return np.vstack((x,y)),z

'Parameters'
N=600
delta=0.05
T=10

m=7 # Disturbance 

alpha1=0.1
alpha2=1
alpha3=5
alpha4=10

'Birkhoff-Rott'
p,q,u1=Bessel(N,delta,1,m,T,0)  
plt.figure(1)
plt.plot(p,q,'b',lw=3)

#'Bessel Function of the First Kind'     
#x,y,u3=Bessel(N,delta,0,m,T,alpha2)
#plt.figure(3)
#plt.title('alpha=1')
#plt.plot(x,y,'k',lw=3)


#'Bessel Function of the First Kind'     
#x,y,u4=Bessel(N,delta,0,m,T,alpha3)
#plt.figure(4)
#plt.title('alpha=5')
#plt.plot(x,y,'k',lw=3)
#
#'Bessel Function of the First Kind'     
#x,y,u5=Bessel(N,delta,0,m,T,alpha4)
#plt.figure(5)
#plt.title('alpha=10')
#plt.plot(x,y,'k',lw=3)
#
#'Bessel Function of the First Kind'     
#x,y,u2=Bessel(N,delta,0,m,T,alpha1)
#plt.figure(2)
#plt.title('alpha=0.1')
#plt.plot(x,y,'k',lw=3)