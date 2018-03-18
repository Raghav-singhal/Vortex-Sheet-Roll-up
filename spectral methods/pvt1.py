# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:04:28 2015

@author: raghavsinghal
"""
from scipy.fftpack import dct
from numba import jit,float64
import numpy as np
from math import sin,cos,pi,sqrt,cosh,sinh
import matplotlib.pyplot as plt

@jit(float64(float64[:]))
def weight(z):
    return np.ones(len(z))

@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def thickness(l,x,y,b):
    N=len(x)
    xnew = x[0:N-1] - x[1:N]
    ynew = y[0:N-1] - y[1:N]
    lnew = np.sqrt(xnew*xnew + ynew*ynew)
    b=b*l/lnew
    return b

@jit(float64(float64[:],float64[:],float64,))
def f(x,y,delta):
    N=len(x)
    n=float(N)
    dx=np.zeros(N)
    dy=np.zeros(N)
    for j in range(N):
        for k in range(N):
            xnum = sinh(2*pi*(y[j]-y[k]))
            ynum = sin(2*pi*(x[j]-x[k]))
            denom = cosh(2*pi*(y[j]-y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2
            dx[j] += xnum/denom
            dy[j] += ynum/denom
    return np.vstack((-0.5*dx/n,0.5*dy/n))

@jit(float64(float64[:],float64[:],float64[:],float64,float64))
def RK4(x,y,z,T,delta):
    dt=0.01
    Finaltime=int(T/dt)
    N=len(x)
    #b=0.1*np.ones(N-1)
    b=0.1*np.sin(pi*z[0:N-1])
    for t in range(Finaltime):
        xmid = (x[0:N-1] - x[1:N])
        ymid = (y[0:N-1] - y[1:N])
        l = np.sqrt(xmid*xmid + ymid*ymid)
        
        F1= f(x,y,delta)
        xk1,yk1 = dt*F1[0],dt*F1[1]
        
        F2= f(x + 0.5*xk1,y + 0.5*yk1, delta)
        xk2 = dt*F2[0]
        yk2 = dt*F2[1]
        
        F3=f(x + 0.5*xk2,y + 0.5*yk2,delta)
        xk3 = dt*F3[0]
        yk3 = dt*F3[1]
        
        F4=f(x + xk3,y+ yk3,delta)
        xk4 = dt*F4[0]
        yk4 = dt*F4[1]
        
        #F5=f(xk4,yk4,delta)
        x = x + (1./6.)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        y = y + (1./6.)*(yk1 + 2*yk2 + 2*yk3 + yk4)
        
        b=thickness(l,x,y,b)
        
        if t%50==0:
            print (Finaltime-t)
            
    return x,y,b
        
@jit(float64(float64,float64,float64))
def pvt1(N,delta,T):
    n=float(N)
    z = np.arange(N)/(n)
    y = -0.01*np.sin(2*pi*z)
    x = z + 0.01*np.sin(2*pi*z)
    
    x,y,b = RK4(x,y,z,T,delta) # x and y after integrating, b is the updated thickness
    p,q = interpolate(x,y,z)
    
    x1,y1,x2,y2,a=curve(x,y,z,b)
    
    plt.figure(1)    
    #plt.plot(x,y,'b')
    #plt.plot(p,q,'r')
    
    plt.figure(2)
    plt.title("evolution of curve")
    plt.plot(p,q,'r')
    plt.plot(x1,y1,'k')
    plt.plot(x2,y2,'k')
    
    
    
    return a


@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def curve(x,y,z,b):
    N=len(x)    
    n=float(N)
    xmid = x
    ymid = y
    zmid = z
    
    #xmid=(x[0:N-1] + x[1:N])*0.5
    #ymid=(y[0:N-1] + y[1:N])*0.5
    #zmid=(z[0:N-1] + z[1:N])*0.5
    
    bhat=dct(b)/(len(b)-1.)
    bhat[0]*=0.5
    #Q=len(b)
    
    b=np.ones(N)
    
    for i  in range(N):
        #b[i]=T(zmid[i]*0.5,bhat).real
        b[i]=T(zmid[i]*0.5,bhat).real
            
    #mode=0
    xhat=np.fft.rfft(x)
    yhat=np.fft.rfft(y)

    ik1=1j*np.hstack((np.arange(0,N/2 +1)))#,np.arange(-N/2 +1,0)))
   
    
    dyspec=ik1*yhat
    dxspec=ik1*xhat
    
    dy2spec=ik1*ik1*yhat
    dx2spec=ik1*ik1*xhat

    dy=np.fft.irfft(dyspec)
    dx=np.fft.irfft(dxspec)
  
    dy2=np.fft.irfft(dy2spec)
    dx2=np.fft.irfft(dx2spec)
    
    
    plt.figure(8)
    #plt.scatter(np.arange(len(dydx)),dydx)
    plt.title("log plot of fourier coefficients of $x_\Gamma$")
    plt.xlabel("wavenumber")
    plt.ylabel("$log|\hat{x}_k|$")
    plt.plot(np.arange(len(dxspec)),np.log(np.abs(dxspec)))
    
    plt.figure(9)
    plt.title("evolution of $y_{\Gamma \Gamma}$")
    plt.plot(z[0:len(dy2)],dy2)
    
    plt.figure(10)
    plt.title("evolution of $x_{\Gamma \Gamma}$")
    plt.scatter(z[0:len(dx2)],dx2)
    
    slope=(ymid[1:N-1]-ymid[0:N-2])/(xmid[1:N-1]-xmid[0:N-2]) 

    fslope=np.fft.rfft(slope)
    
    slope=np.fft.irfft(fslope)
    
    
    x1=np.ones(N-2)
    y1=np.ones(N-2)
    x2=np.ones(N-2)
    y2=np.ones(N-2)
    for i in range(N-2):                
        if (xmid[i+1]-xmid[i])<=0 and (ymid[i+1]-ymid[i])>0:
            y1[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
            x1[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
            y2[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
            x2[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        
        elif (xmid[i+1]-xmid[i])<=0 and (ymid[i+1]-ymid[i])<=0:
            y1[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
            x1[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
            y2[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
            x2[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        
        elif (xmid[i+1]-xmid[i])>=0 and (ymid[i+1]-ymid[i])<0:
            y1[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
            x1[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. +slope[i]**2))
            
            y2[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
            x2[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        
        else:
            y1[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
            x1[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
            y2[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
            x2[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
    
    plt.figure(3)
    plt.title("real slope")
    plt.plot(zmid[0:len(slope)],slope,'bo')
    
    #plt.figure(5)
    #plt.title("spectral slope")
    #plt.scatter(np.arange((N-2)/2 + 1),np.fft.rfft(slope))
    
    #ik=1j*np.hstack((np.arange(0,(N)/2 +1),np.arange(-(N)/2 +1,0)))


    #curvaturehat=np.fft.fft(slope)
    #curvaturehat=ik*curvaturehat
    #curvature=np.fft.ifft(curvaturehat)
    
    #plt.figure(6)
    #plt.title("real curvature")
    #plt.scatter(zmid[0:N],curvature)
    
    #plt.figure(7)
    #plt.title("spectral curvature")
    #plt.plot(np.arange(N),curvaturehat)

    
    return x1,y1,x2,y2,len(dy)
           
@jit(float64(float64[:],float64[:],float64[:]))
def interpolate(x,y,z):  
    N=len(x)
    n=float(N)
    p = (x-z) + 1j*y

    phat = np.fft.fft(p)/n
    phat[0]*=0.5
    
    a=np.ones(1000,dtype=complex)
    J=np.arange(1000)/1000.
    for i in range(1000):
        a[i]=T(J[i],phat) + J[i]
    
    xhat=np.fft.rfft(x-z)
    yhat=np.fft.rfft(y)
    
    plt.figure(4)
    plt.title("log plot of fourier coefficients of $x - \Gamma$")
    plt.xlabel("wavenumber")
    plt.ylabel("$log|\hat{X}_k|$")
        
    plt.plot(np.arange(len(xhat)),np.log(np.abs(xhat)))

    plt.figure(11)
    plt.title("log plot of fourier coefficients of $y$")
    plt.xlabel("wavenumber")
    plt.ylabel("$log|\hat{y}_k|$")
    
    plt.plot(np.arange(len(xhat)),np.log(np.abs(yhat)))
    
    #plt.plot(np.arange(len(phat)),np.log(np.abs(np.fft.rfft(p))),'bo')
        
    return a.real,a.imag

@jit(float64(float64,float64[:])) #interpolating polynomial
def T(z,phat):
    M=int(len(phat)/2)
    I=complex(0.,0.)
    for k in range(-M + 1, M):
        I+=phat[k]*(cos(2*pi*k*z) + 1j*sin(2*pi*k*z))
    return I

    
        
        