# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:01:56 2015

@author: raghavsinghal
"""
from numba import jit,float64
import numpy as np
from math import sin,cos,pi,sqrt,cosh,sinh
import matplotlib.pyplot as plt

@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def thickness(l,x,y,b):
    N=len(x)
    x2mid=(x[0:N-1] - x[1:N])
    y2mid=(y[0:N-1] - y[1:N])
    lnew=np.sqrt(x2mid**2 + y2mid**2)
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
            if k!=j:
               xnum = sinh(2*pi*(y[j]-y[k]))
               ynum = sin(2*pi*(x[j]-x[k]))
               denom = cosh(2*pi*(y[j]-y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2
               dx[j] += xnum/denom
               dy[j] += ynum/denom
    return np.vstack((-0.5*dx/n,0.5*dy/n))

@jit(float64(float64[:],float64[:],float64[:],float64,float64))
def RK4(x,y,z,T,delta):
    N=len(x)
    dt=0.002
    Finaltime=int(T/dt)
    b=0.1*np.ones(N-1)#*(np.sin(pi*z[0:len(x)-1]))

#    xdata,ydata=x,y 
#    curvaturedata=plots(x,y,z)
#    x1data,y1data,x2data,y2data=curve(x,y,b)
#    zdata=z
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
        
        x = x + (1./6.)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        y = y + (1./6.)*(yk1 + 2*yk2 + 2*yk3 + yk4)
        
#        curvature1=plots(x,y,z)
#        x1,y1,x2,y2=curve(x,y,b)
#        curvaturedata=np.append(curvaturedata,curvature1)
#        #strengthdata=np.append(strengthdata,strength1)
#        #slopedata=np.append(slopedata,slope1)
#        
#        zdata=np.append(zdata,z)
#        x1data=np.append(x1data,x1)   
#        y1data=np.append(y1data,y1)
#        x2data=np.append(x2data,x2)
#        y2data=np.append(y2data,y2)
#        xdata=np.append(xdata,x)        
#        ydata=np.append(ydata,y)        
#        
        xhat=np.fft.rfft(x-z)
        yhat=np.fft.rfft(y)
        

        for i in range(N/2 + 1):
            if  abs(xhat[i])<1e-13:
                xhat[i]=0.0
            if abs(yhat[i])<1e-13:
                yhat[i]=0.
                
        x=np.fft.irfft(xhat) + z
        y=np.fft.irfft(yhat)
        b=thickness(l,x,y,b)

        if t%500==0:
            print (Finaltime-t)
        
#    np.savetxt("D02curvature.txt",curvaturedata)
#    np.savetxt("D02xdata.txt",xdata)
#    np.savetxt("D02ydata.txt",ydata)
#    np.savetxt("D02x1data.txt",x1data)
#    np.savetxt("D02y1data.txt",y1data)
#    np.savetxt("D02x2data.txt",x2data)
#    np.savetxt("D02y2data.txt",y2data)


    #np.savetxt("strengthD01.txt",strengthdata)
    #np.savetxt("slopeD01.txt",slopedata)
    #np.savetxt("z.txt",zdata)       
    return x,y,b
        
@jit(float64(float64,float64,float64))
def singular(N,delta,T):
    n=float(N)
    z =np.arange(N)/(n)
    y = -0.01*np.sin(2*pi*z)
    x = z + 0.01*np.sin(2*pi*z)
    
    x,y,b = RK4(x,y,z,T,delta) # x and y after integrating, b is the updated thickness
    #p,q = interpolate(x,y,z)
    
    a=plots(x,y,z)
    x1,y1,x2,y2=curve(x,y,b)
    
    plt.figure(1)
    plt.title("Evolution of curve",fontsize=20)
    plt.plot(x,y,'r',lw=2)
    plt.plot(x1,y1,'ko',lw=2)
    plt.plot(x2,y2,'ko',lw=2)
    #plt.axis([0,1,-0.2,0.2])
    plt.xlabel('x',fontsize=20)
    plt.ylabel("y", fontsize=20)
    #plt.savefig("D005CURVET052.pdf")
    
    


@jit(float64(float64[:],float64[:],float64[:]))
def plots(x,y,z):
    N=len(x)    

    xhat=np.fft.rfft(x-z)
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
    
    
    plt.figure(2)
    plt.title("log plot of fourier coefficients of $x_\Gamma$")
    plt.xlabel("wavenumber")
    plt.ylabel("$log|\hat{x}_k|$")
    plt.plot(np.arange(len(dxspec)),np.log(np.abs(dxspec)))
    
    plt.figure(3)
    plt.title("evolution of $y_{\Gamma \Gamma}$")
    plt.plot(z[0:len(dy2)],dy2)
    
    plt.figure(4)
    plt.title("evolution of $x_{\Gamma \Gamma}$")
    plt.scatter(z[0:len(dx2)],dx2)
#    
#    slope=(y[1:N-1]-y[0:N-2])/(x[1:N-1]-x[0:N-2]) 
#        
#    plt.figure(5)
#    plt.title("slope",fontsize=20)
#    plt.scatter(z[0:len(slope)],slope,lw=2)
#    plt.axis([0,1,-150,250])
#    plt.xlabel('$\Gamma$',fontsize=20)
#    plt.ylabel("dy/dx", fontsize=20)
#    plt.savefig("D005slopeT052.pdf")

    
    curvature=np.zeros(N)
    strength=np.zeros(N)
    for i in range(N):
        if z[i]>0.5:
            strength[i]=1.0/(((x[i]-x[i-1])/(z[i]-z[i-1]))**2 + ((y[i]-y[i-1])/(z[i]-z[i-1]))**2)**0.5
            curvature[i]=(((x[i]-x[i-1])/(z[i]-z[i-1]))*(((y[i]-y[i-1])/(z[i]-z[i-1])-(y[i-1]-y[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])) - ((y[i]-y[i-1])/(z[i]-z[i-1]))*((x[i]-x[i-1])/(z[i]-z[i-1]) - (x[i-1]-x[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1]))*strength[i]**3
        elif z[i]<0.5:
            strength[i]=1.0/(((x[i+1]-x[i])/(z[i+1]-z[i]))**2 + ((y[i+1]-y[i])/(z[i+1]-z[i]))**2)**0.5
            curvature[i]=(((x[i+1]-x[i])/(z[i+1]-z[i]))*((y[i+1]-y[i])/(z[i+1]-z[i])-(y[i]-y[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i]) - ((y[i+1]-y[i])/(z[i+1]-z[i]))*((x[i+1]-x[i])/(z[i+1]-z[i]) - (x[i]-x[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i]))*(strength[i]**3)
          
    for i in range(N):
        if z[i]==0.5:
            strength[i]=(strength[i-1] + strength[i+1])*0.5
            curvature[i]=(curvature[i-1] + curvature[i+1])*0.5
        elif z[i]==0:
            curvature[i]=curvature[N-1]
#    
#    plt.figure(12,figsize=(13,4))
#    plt.title("curvature",fontsize=20)
#    plt.plot(z,curvature,'k',lw=2)
#    plt.axis([0,1,-60,80])
#    plt.xlabel('$\Gamma$',fontsize=20)
#    plt.ylabel("$\kappa$", fontsize=20)
#    plt.savefig("D005curvatureT052.pdf")
#
# 
#    plt.figure(13)
#    plt.title("strength",fontsize=20)
#    plt.axis([0,1,0.5,3.8])
#    plt.plot(z,strength,'k',lw=2)
#    plt.xlabel('$\Gamma$',fontsize=20)
#    plt.ylabel("$\sigma$", fontsize=20)
#    plt.savefig("D005strengthT052.pdf")


    return curvature
    
#@jit(float64(float64[:],float64[:],float64[:]))
#def interpolate(x,y,z):  
#    N=len(x)
#    n=float(N)
#    p = (x-z) + 1j*y
#
#    phat = np.fft.fft(p)/n
#    phat[0]*=0.5
#    
#    a=np.ones(N,dtype=complex)
#    J=np.arange(N)/n
#    for i in range(N):
#        a[i]=T(J[i],phat) + J[i]
#    
#    xhat=np.fft.rfft(x-z)
#    yhat=np.fft.rfft(y)
#    
#    plt.figure(9)
#    plt.title("plot of fourier coefficients of x - $\Gamma$")
#    plt.xlabel("wavenumber")
#    plt.ylabel("$|\hat{X}_k|$")    
#    plt.plot(np.arange(len(xhat)),(np.abs(xhat)))
#
#    plt.figure(10)
#    plt.title("plot of fourier coefficients of $y$")
#    plt.xlabel("wavenumber")
#    plt.ylabel("$|\hat{y}_k|$")
#    plt.plot(np.arange(len(yhat)),(np.abs(yhat)))
#    
#    
#    phatAM=np.log(np.abs(np.fft.rfft(p)))
#    plt.figure(11)
#    plt.title("log plot of $\hat{p}_k$")
#    plt.plot(np.arange(len(phatAM)),phatAM,'b')
#        
#    return a.real,a.imag

#@jit(float64(float64,float64[:])) #interpolating polynomial
#def T(z,phat):
#    M=int(len(phat)/2)
#    I=complex(0.,0.)
#    for k in range(-M + 1, M):
#        I+=phat[k]*(cos(2*pi*k*z) + 1j*sin(2*pi*k*z))
#    return I

@jit(float64(float64[:],float64[:],float64[:]))
def curve(x,y,b): # x and y after interpolation
    N=len(x)
    xmid=(x[0:N-1] + x[1:N])*0.5
    ymid=(y[0:N-1] + y[1:N])*0.5
    slope=(ymid[1:N-1]-ymid[0:N-2])/(xmid[1:N-1]-xmid[0:N-2]) 
    
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

        
    return x1,y1,x2,y2
        