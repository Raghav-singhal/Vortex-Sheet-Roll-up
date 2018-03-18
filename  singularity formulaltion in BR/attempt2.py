# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:44:18 2015

@author: raghavsinghal
"""

from numba import jit,float64
import numpy as np
from math import sin,cos,pi,sqrt,cosh,sinh
import matplotlib.pyplot as plt


@jit(float64(float64,float64,float64))
def at2(N,delta,T):
    n=float(N)
    z =np.arange(N)/(n-1.)
    
    z=np.cos(pi*z)   
    
    y = -0.01*np.sin(pi*z)
    x = z + 0.01*np.sin(pi*z)
    
    x,y,b= RK4(x,y,z,T,delta) # x and y after integrating, b is the updated thickness
    
    c=curvature(x,y,z,1)

    x1,y1,x2,y2=curve(x,y,b)

    plt.figure(1)
    plt.title("Evolution of curve",fontsize=20)
    plt.plot(x,y,'b',lw=1)
    #plt.plot(x1,y1,'bo',lw=2)
    #plt.plot(x2,y2,'bo',lw=2)

    
    
    #plt.axis([0,1,-0.3,0.3])
    plt.xlabel('x',fontsize=20)
    plt.ylabel("y", fontsize=20)
    #plt.savefig("D005CURVET052.pdf")

def rotation(x,y,z,maxc,minc):
    A=np.array([[z[maxc+1]**3,z[maxc+1]],[z[minc-1]**3,z[minc-1]]])
    u=np.array([x[maxc+1],x[minc-1]])
    v=np.array([y[maxc+1],y[minc-1]])
    
    a1=np.linalg.solve(A,u)
    a2=np.linalg.solve(A,v)
    x[minc:maxc]=a1[0]*z[minc:maxc]**3 + a1[1]*z[minc:maxc]
    y[minc:maxc]=a2[0]*z[minc:maxc]**3 + a2[1]*z[minc:maxc]
    return x,y
    
    
@jit(float64(float64[:],float64[:],float64,))
def f(x,y,delta):
    N=len(x)
    n=float(N)
    dx=np.zeros(N)
    dy=np.zeros(N)

    for j in range(N):
        for k in range(N):
            if k!=j:
               xnum = sinh(pi*(y[j]-y[k]))
               ynum = sin(pi*(x[j]-x[k]))
               denom = cosh(pi*(y[j]-y[k])) - cos(pi*(x[j]-x[k])) + delta**2
               dx[j] += xnum/denom
               dy[j] += ynum/denom
    return np.vstack((-0.5*dx/n,0.5*dy/n))

@jit(float64(float64[:],float64[:],float64,float64,float64))
def shock(x,y,delta,maxc,minc):
    N=len(x)
    n=float(N)
    dx=np.zeros(N)
    dy=np.zeros(N)

    maxc=int(maxc)
    minc=int(minc)
    
    
    for j in range(N):
        if minc<=j<=maxc:
            dx[j]=0
            dy[j]=0
        else:
           for k in range(N):
               if k!=j:
                  xnum = sinh(pi*(y[j]-y[k]))
                  ynum = sin(pi*(x[j]-x[k]))
                  denom = cosh(pi*(y[j]-y[k])) - cos(pi*(x[j]-x[k])) + delta**2
                  dx[j] += xnum/denom
                  dy[j] += ynum/denom
    #dx[minc:maxc+1]=np.zeros(maxc+1-minc)
    #dy[minc:maxc+1]=np.zeros(maxc+1-minc)
    #for i in range(min,max):
     
    #print max        
    return np.vstack((-0.5*dx/n,0.5*dy/n))
        
@jit(float64(float64[:],float64[:],float64[:],float64,float64))
def RK4(x,y,z,T,delta):
    N=len(x)
    dt=0.01
    Finaltime=int(T/dt)
    b=0.1*np.ones(N-1)#*(np.sin(pi*z[0:len(x)-1]))

    
    t=0
    xpptol=0
    while t<=Finaltime:
        if xpptol<2 : #shock tolerance still need to determine a better way to calculate
           xmid = (x[0:N-1] - x[1:N])
           ymid = (y[0:N-1] - y[1:N])
           l = np.sqrt(xmid*xmid + ymid*ymid)
            
           #xhat=np.fft.rfft(x-z)
           #ik=1j*np.arange(N/2 +1)
           #ddx1spec=ik*ik*xhat
           #ddx1=np.fft.irfft(ddx1spec)
          
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
        
           x1,y1,x2,y2=curve(x,y,b)

           
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
        
           xhat=np.fft.rfft(x-z)
           ik=1j*np.arange(N/2 +1)
           ddxspec=ik*ik*xhat
           
           ddx=np.fft.irfft(ddxspec)
           xpptol=np.max(ddx)-np.min(ddx)
           
           
        
           if (Finaltime-t)%10==0:# and maxcurvaturetol>2 :
              print Finaltime-t
        
           if xpptol>3:
              print 'shock',xpptol
              break
          
           maxcurve=np.argmax(ddx)
           mincurve=np.argmin(ddx)
           t=t+1
    """
    change loops here . 
    """
        
    plt.figure(15)
    plt.title('xpp at critical time')
    plt.plot(z[mincurve:maxcurve],ddx[mincurve:maxcurve],'bo')
    
    for k in range(t+1,Finaltime):
        xmid = (x[0:N-1] - x[1:N])
        ymid = (y[0:N-1] - y[1:N])
        l = np.sqrt(xmid*xmid + ymid*ymid)
          
        F1= shock(x,y,delta,maxcurve,mincurve)
        xk1,yk1 = dt*F1[0],dt*F1[1]
          
        F2= shock(x + 0.5*xk1,y + 0.5*yk1, delta,maxcurve,mincurve)
        xk2 = dt*F2[0]
        yk2 = dt*F2[1]
        
        F3=shock(x + 0.5*xk2,y + 0.5*yk2,delta,maxcurve,mincurve)
        xk3 = dt*F3[0]
        yk3 = dt*F3[1]
        
        F4=shock(x + xk3,y+ yk3,delta,maxcurve,mincurve)
        xk4 = dt*F4[0]
        yk4 = dt*F4[1]
        
        x = x + (1./6.)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        y = y + (1./6.)*(yk1 + 2*yk2 + 2*yk3 + yk4)
        
        #x,y=rotation(x,y,z,maxcurve,mincurve)
        
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
        
        xhat=np.fft.rfft(x)
        #yhat=np.fft.rfft(y)
        
        ik=1j*np.arange(N/2+1)

        dx=ik*ik*xhat
        #dy=ik*ik*yhat

        dx=np.fft.irfft(dx)
        #dy=np.fft.irfft(dy)
        #dxtol=np.max(dx)-np.min(dx)
        #ypp=dy
        mincurve=np.argmin(dx)
        maxcurve=np.argmax(dx)
 
        
        print 'loop changed'
    return x,y,b
         

@jit(float64(float64[:],float64[:],float64[:]))
def curve(x,y,b): # x and y after interpolation
    N=len(x)
    xmid=(x[0:N-1] + x[1:N])*0.5
    ymid=(y[0:N-1] + y[1:N])*0.5
    slope=(ymid[1:N-1]-ymid[0:N-2])/(xmid[1:N-1]-xmid[0:N-2]) 
    
    z=np.arange(N)/float(N)
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
    plt.figure(5)
    plt.title("slope",fontsize=20)
    plt.plot(z[0:len(slope)],slope,lw=2)
#    plt.axis([0,1,-150,250])
    plt.xlabel('$\Gamma$',fontsize=20)
    plt.ylabel("dy/dx", fontsize=20)
#    plt.savefig("D005slopeT052.pdf")        
    return x1,y1,x2,y2
    
@jit(float64(float64[:],float64[:],float64[:],float64))
def curvature(x,y,z,Plot):
    N=len(x)    

    xhat=np.fft.rfft(x-z)
    yhat=np.fft.rfft(y)

    ik1=1j*np.hstack((np.arange(0,N/2 +1)))#,np.arange(-N/2 +1,0)))
    
    dy2spec=ik1*ik1*yhat
    dx2spec=ik1*ik1*xhat
  
    dy2=np.fft.irfft(dy2spec)
    dx2=np.fft.irfft(dx2spec)
    
    
    
    if Plot==1:
       plt.figure(3)
       plt.title("evolution of $y_{\Gamma \Gamma}$")
       plt.scatter(z[0:len(dy2)],dy2)
    
       plt.figure(4)
       plt.title("evolution of $x_{\Gamma \Gamma}$")
       #plt.axis([0.3,0.7,-1.75,1.75])
       plt.scatter(z[0:len(dx2)],dx2)
    
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
   
    if Plot==1:
        plt.figure(12)#,figsize=(13,4))
        plt.title("curvature",fontsize=20)
        plt.plot(z,curvature,'k',lw=2)
        #plt.axis([0,1,-20,20])
        plt.xlabel('$\Gamma$',fontsize=20)
        plt.ylabel("$\kappa$", fontsize=20)
        #plt.savefig("D005curvatureT052.pdf")

 
        #plt.figure(13)
        #plt.title("strength",fontsize=20)
        #plt.axis([0,1,0.5,3.8])
        #plt.plot(z,strength,'k',lw=2)
        #plt.xlabel('$\Gamma$',fontsize=20)
        #plt.ylabel("$\sigma$", fontsize=20)
        #plt.savefig("D005strengthT052.pdf")
    
    return curvature
    
@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def thickness(l,x,y,b):
    N=len(x)
    x2mid=(x[0:N-1] - x[1:N])
    y2mid=(y[0:N-1] - y[1:N])
    lnew=np.sqrt(x2mid**2 + y2mid**2)
    b=b*l/lnew
    return b