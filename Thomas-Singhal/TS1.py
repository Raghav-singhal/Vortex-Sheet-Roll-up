# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:11:58 2015

@author: raghavsinghal
"""
from numba import jit,float64,int64
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from math import log,sinh,atan2,cos,pi,sqrt
#from scipy.special import k1

'Parameters'
N=400
alpha=0.1

T=1
gamma=1
delta=0.25

def TS1(N,alpha,gamma,delta,T):
    z1=np.arange(N)/float(N)
    z2=np.arange(N)/float(N)

    x1=0.25*np.cos(2*pi*z1)
    y1=0.25*np.sin(2*pi*z1)
    
    x2=0.25*np.cos(2*pi*z2) 
    y2=0.25*np.sin(2*pi*z2) 

    r1=np.vstack((x1,y1))
    r2=np.vstack((x2,y2))
    
    x1,y1,x2,y2=RKF(r1,r2,z1,z2,T,alpha,gamma,delta)
    
    x=np.append(x1,x2)
    y=np.append(y1,y2 + 4)
    
    plt.figure(1)    
#    plt.figure(figsize=(9,13))
    plt.scatter(x,y)



def k1(r):
    r=float(r)
    a=atan2(4,log(sinh(r)))
    b=0.534 - 0.6*cos(a) -0.068*cos(2*a) + 0.125*cos(3*a) + 0.032*cos(4*a)-0.032*cos(5*a)
    return b/r

#@jit(float64(float64[:,:],float64[:,:],float64[:],float64[:],int64,float64,float64,float64))
def RKF(r1,r2,z1,z2,T,alpha,gamma,delta):
    etol=1e-3
    
    dt=0.05
    
    y=np.vstack((r1,r2))    
        
    u=y
    N=np.array([len(r1[1,:])])    
    dT=np.array([dt])
    time=np.zeros(1)

    K1=np.zeros((4,1))
    K2=np.zeros((4,1))
    K3=np.zeros((4,1))
    K4=np.zeros((4,1))
    K5=np.zeros((4,1))
    K6=np.zeros((4,1))
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
    e3=2.
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
              K1=dt*f(t,y,alpha,gamma,delta)

              K2=dt*f(t + 0.25*dt, y + a2*K1,alpha,gamma,delta)

              K3=dt*f(t + (3./8.)*dt, y + b2*K1 + b3*K2,alpha,gamma,delta)

              K4=dt*f(t + (12./13.)*dt, y + c2*K1 + c3*K2 + c4*K3,alpha,gamma,delta)
              
              K5=dt*f(t + dt, y + d2*K1 + d3*K2 + d4*K3 + d5*K4,alpha,gamma,delta)
              
              K6=dt*f(t + 0.5*dt, y + e2*K1 + e3*K2 + e4*K3 + e5*K4 + e6*K5,alpha,gamma,delta)
              
              error=np.max(abs(r1*K1 + r3*K3 + r4*K4 + r5*K5 + r6*K6)/dt)
              
              if (error)<etol or dt<1e-3:
                  
                 y=y + a11*K1 + a13*K3 + a14*K4 + a15*K5 + a16*K6
                        
                 #y,z1,z2=meshrefinement(y,z1,z2)
                 
                 u=np.hstack((u,y))
                 time=np.append(time,t)
                 N=np.append(N,len(y[0,:]))
                 dT=np.append(dT,dt)
                 if n%1==0:  
                    print (T-t),len(y[0,:]),len(y[3,:])
                 
                 t=t+dt
                 n+=1              
                 
                 
              else:
                 dt=0.84*dt*((etol/(error))**(0.25))
                 print (dt)
    
    
    np.savetxt('X1dataBessel.txt',u[0,:])
    np.savetxt('Y1dataBessel.txt',u[1,:])
    np.savetxt('X2dataBessel.txt',u[2,:])
    np.savetxt('Y2dataBessel.txt',u[3,:])    
    
    np.savetxt('NdataBessel.txt',N)
    np.savetxt('dTdataBessel.txt.',dT)
    np.savetxt('NumberOfIterationBessel.txt',np.array([n-1]))
    #print 'saved'
    
    return y[0,:],y[1,:],y[2,:],y[3,:]      
      
@jit(float64(float64,float64[:,:],float64,float64,float64))
def f(t,X,alpha,gamma,delta):
    x1=X[0,:]
    y1=X[1,:]
    x2=X[2,:]
    y2=X[3,:]
    
    kd=gamma*sqrt(1+delta)
    
    q1=1
    q2=0
    
    a1=q1*delta/(1 + delta)
    a2=q2/(1 + delta)
    a3=-q1/(1 + delta)
    a4=q2/(1 + delta)
    
    b1=q1*delta/(1 + delta)
    b2=q2/(1 + delta)
    b3=q1*delta/(1 + delta)
    b4=-q2*delta/(1 + delta)
    
    N,n=len(x1),float(len(x1))
    
    dx1,dy1,dx2,dy2=np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
    
    for j in range(N):
        for k in range(j,N):
            num11=-(a1 + a2)*(y1[j] - y1[k])
            num21=(a1 + a2)*(x1[j] -x1[k])
            denom=(y1[j] - y1[k])**2 + (x1[j] -x1[k])**2 + alpha**2
            
            num21=-(a3 + a4)*kd*(y1[j] - y1[k])*k1(kd*sqrt((y1[j] - y1[k])**2 + (x1[j] - x1[k])**2 + alpha**2))
            num22=(a3 + a4)*kd*(x1[j] - x1[k])*k1(kd*sqrt((y1[j] - y1[k])**2 + (x1[j] - x1[k])**2 + alpha**2))
            
            dx1[j]+=num11/denom + num21/sqrt(denom)
            dy1[j]+=num21/denom + num22/sqrt(denom)
                
            dx1[k]+=-(num11/denom + num21/sqrt(denom))
            dy1[k]+=-(num21/denom + num22/sqrt(denom))

            num11=-(b1 + b2)*(y2[j] - y2[k])
            num21=(b1 + b2)*(x2[j] -x2[k])
            denom=(y2[j] - y2[k])**2 + (x2[j] -x2[k])**2 + alpha**2
            
            num21=-(b3 + b4)*kd*(y2[j] - y2[k])*k1(kd*sqrt((y2[j] - y2[k])**2 + (x2[j] - x2[k])**2 + alpha**2))
            num22=(b3 + b4)*kd*(x2[j] - x2[k])*k1(kd*sqrt((y2[j] - y2[k])**2 + (x2[j] - x2[k])**2 + alpha**2))
            
            dx2[j]+=num11/denom + num21/sqrt(denom)
            dy2[j]+=num21/denom + num22/sqrt(denom)
                
            dx2[k]+=-(num11/denom + num21/sqrt(denom))
            dy2[k]+=-(num21/denom + num22/sqrt(denom))

    dr1=np.vstack((dx1,dy1))
    dr2=np.vstack((dx2,dy2))
    return np.vstack((0.5*dr1/n,0.5*dr2/n))


@jit(float64(float64[:,:],float64[:],float64[:]))
def meshrefinement(u,z1,z2):
    x=u[0,:]
    y=u[1,:]
    
    a,b=u[2,:],u[3,:]    
    
    t=0    
    etolmin=0.005
    etolmax=0.02
    
    while t<len(x)-1:
        if sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)>=etolmax:
            xtck=interpolate.splrep(z1,x,k=3)
            ytck=interpolate.splrep(z1,y,k=3)
            
            znew=0.5*(z1[t] + z1[t+1])
            
            xnew=interpolate.splev(znew,xtck)
            ynew=interpolate.splev(znew,ytck)
            
            x=np.insert(x,t+1,xnew)
            y=np.insert(y,t+1,ynew)
            z1=np.insert(z1,t+1,znew)

        if sqrt((a[t] - a[t+1])**2 + (b[t] - b[t+1])**2)>=etolmax:
            atck=interpolate.splrep(z2,x,k=3)
            btck=interpolate.splrep(z2,y,k=3)
            
            znew=0.5*(z2[t] + z2[t+1])
            
            anew=interpolate.splev(znew,atck)
            bnew=interpolate.splev(znew,btck)
            
            a=np.insert(x,t+1,anew)
            b=np.insert(y,t+1,bnew)
            z2=np.insert(z2,t+1,znew)
            
        elif sqrt((x[t] - x[t+1])**2 + (y[t] - y[t+1])**2)<=etolmin:
            
            x=np.delete(x,t+1)
            y=np.delete(y,t+1)
            z1=np.delete(z1,t+1)

        elif sqrt((a[t] - a[t+1])**2 + (b[t] - b[t+1])**2)<=etolmin:
            
            a=np.delete(a,t+1)
            b=np.delete(b,t+1)
            z2=np.delete(z2,t+1)
        t+=1
    
    Q=max(len(a),len(x))
    xtck=interpolate.splrep(z1,x,k=3)
    ytck=interpolate.splrep(z1,y,k=3)
    z1=np.arange(Q)/float(Q)
    
    x=interpolate.splev(z1,xtck)
    y=interpolate.splev(z1,ytck)

    atck=interpolate.splrep(z2,a,k=3)
    btck=interpolate.splrep(z2,b,k=3)
    z2=np.arange(Q)/float(Q)
    
    a=interpolate.splev(z2,xtck)
    b=interpolate.splev(z2,ytck)

    r1=np.vstack((x,y))
    r2=np.vstack((a,b))

    return np.vstack((r1,r2)),z1,z2
    
TS1(N,alpha,gamma,delta,T)
    