# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:50:04 2015

@author: raghavsinghal
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit,complex64,float64,int64
from math import pi 

@jit(complex64(float64,complex64[:,:],float64,float64,float64,float64))
def f(t,u,a,b,g,e):
    dx=np.zeros((18,1),dtype='complex')
    
    
    ua=u[0,0]
    ub=u[1,0]
    ug=u[2,0]

    va=u[3,0]
    vb=u[4,0]
    vg=u[5,0]

    ha=u[6,0]
    hb=u[7,0]
    hg=u[8,0]

    u_a=u[9,0]
    u_b=u[10,0]
    u_g=u[11,0]

    v_a=u[12,0]
    v_b=u[13,0]
    v_g=u[14,0]

    h_a=u[15,0]
    h_b=u[16,0]
    h_g=u[17,0]    
    
    dx[0,0]=va - 1j*a*ha -1j*e*a*u_b*ug
    dx[1,0]=vb - 1j*b*hb - 1j*e*b*u_a*ug
    dx[2,0]=vg - 1j*g*hg - 1j*e*g*ua*ub
    
    dx[3,0]=-ua - 1j*e*(g*u_b*vg - b*ug*v_b)
    dx[4,0]=-ub - 1j*e*(g*vg*u_a - a*v_a*ug)
    dx[5,0]=-ug - 1j*e*(b*ua*vb + a*ub*va)
    
    dx[6,0]=-1j*a*ua - 1j*e*a*(h_b*ug + hg*u_b)
    dx[7,0]=-1j*b*ub - 1j*e*b*(h_a*ug + hg*u_a)
    dx[8,0]=-1j*g*ug - 1j*e*g*(ha*ub + hb*ua)
    
    dx[9,0]=np.conjugate(dx[0,0])
    dx[10,0]=np.conjugate(dx[1,0])
    dx[11,0]=np.conjugate(dx[2,0])
    
    dx[12,0]=np.conjugate(dx[3,0])
    dx[13,0]=np.conjugate(dx[4,0])
    dx[14,0]=np.conjugate(dx[5,0])
    
    dx[15,0]=np.conjugate(dx[6,0])
    dx[16,0]=np.conjugate(dx[7,0])
    dx[17,0]=np.conjugate(dx[8,0])
        
    return dx

@jit(float64(int64,float64,float64,float64,float64))
def Shallow(T,Q1,Q2,Q3,e):
    etol=1e-1
    dt=0.01

    a=1
    b=1
    g=2
    
    #e=0.1
    
    y=np.zeros((18,1),dtype='complex')    
    
    w=np.sqrt(1 + a**2)

    ha=Q1 + 0*1j
    hb=Q2 + 0*1j 
    hg=Q3 + 0*1j
    
    ua=w*ha/(a)
    ub=-w*hb/(b)
    ug=0 + 0*1j
    
    va=ha/(1j*a)
    vb=hb/(1j*b)
    vg=1j*g*hg
     
    y[0,0]=ua
    y[1,0]=ub
    y[2,0]=ug
    
    y[3,0]=va
    y[4,0]=vb
    y[5,0]=vg
    
    y[6,0]=ha
    y[7,0]=hb
    y[8,0]=hg
    
    y[9,0]=np.conjugate(y[0,0])
    y[10,0]=np.conjugate(y[1,0])
    y[11,0]=np.conjugate(y[2,0])
    
    y[12,0]=np.conjugate(y[3,0])
    y[13,0]=np.conjugate(y[4,0])
    y[14,0]=np.conjugate(y[5,0])

    y[15,0]=np.conjugate(y[6,0])
    y[16,0]=np.conjugate(y[7,0])
    y[17,0]=np.conjugate(y[8,0])    
    
    u=y
    time=np.zeros(1)

    k1=np.zeros((18,1),dtype='complex')
    k2=np.zeros((18,1),dtype='complex')
    k3=np.zeros((18,1),dtype='complex')
    k4=np.zeros((18,1),dtype='complex')
    k5=np.zeros((18,1),dtype='complex')
    k6=np.zeros((18,1),dtype='complex')
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
        while t<T and n*dt<T:
              k1=dt*f(t,y,a,b,g,e)

              k2=dt*f(t + 0.25*dt, y + a2*k1,a,b,g,e)

              k3=dt*f(t + (3./8.)*dt, y + b2*k1 + b3*k2,a,b,g,e)

              k4=dt*f(t + (12./13.)*dt, y + c2*k1 + c3*k2 + c4*k3,a,b,g,e)
              
              k5=dt*f(t + dt, y + d2*k1 + d3*k2 + d4*k3 + d5*k4,a,b,g,e)
              
              k6=dt*f(t + 0.5*dt, y + e2*k1 + e3*k2 + e4*k3 + e5*k4 + e6*k5,a,b,g,e)
              
              error=np.max(abs(r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6)/dt)
              
              if (error)<etol or dt<1e-3:
                 y=y + a11*k1 + a13*k3 + a14*k4 + a15*k5 + a16*k6
                 u=np.hstack((u,y))
                 time=np.append(time,t)
                 t=t+dt
                 n+=1
                 if n%1000==0:  
                    print int(n),T-t
                 
              else:
                 dt=0.84*dt*((etol/(error))**(0.25))
                 print dt
                 
    ua=u[0,:]
    ub=u[1,:]
    ug=u[2,:]

    va=u[3,:]
    vb=u[4,:]
    vg=u[5,:]

    ha=u[6,:]
    hb=u[7,:]
    hg=u[8,:]

    qa=np.abs(1j*a*va - ha)
    qb=np.abs(1j*b*vb - hb)
    qg=(1j*g*vg - hg)
    

    return time,np.abs(qg),w,e#,qgint,hg,vg
    
@jit(complex64(complex64[:],float64[:],float64,float64))
def timeaverage(f,time,w,e):
    N=len(time)
    PQ=float(2*pi/(0.1*e))
    fhat=np.zeros(N,dtype='complex')
    grab=np.zeros((N,N))
    for t in range(N):
        a=0
        #b=0
        length=0
        while length<PQ and t+a<N:
            for p in range(a):
             length+=(time[t+p+1] - time[t+p])
            a=a+1
        grab[t,0]=a
        
    for t in range(N):
        fhat[t]=(1/PQ)*np.trapz(f[t:t + grab[t,0]],time[t:t + grab[t,0]])
    return np.abs(fhat)
    
time,qg_wave,w,e=Shallow(100,2,3,5,0.05)
time,qg_nowave,w,e=Shallow(100,0,0,5,0.05)

#time_waves=timeaverage(qg_wave,time,w,e)
#time_nowaves=timeaverage(qg_nowave,time,w,e)

#plt.figure(1)
#plt.title('$q_g$ with waves and no waves')
#plt.plot(time,np.abs(qg_nowave),'ro')
#plt.plot(time,np.abs(qg_wave),'b')
#plt.savefig('qg_wave.pdf')


#plt.figure(2)
#plt.title('Time Average of $q_g$ with waves and no waves ')
#plt.plot(time[:int(len(time)*0.9)],time_waves[:int(len(time)*0.9)],'b')
#plt.plot(time[:int(len(time)*0.9)],time_nowaves[:int(len(time)*0.9)],'r')
#plt.savefig('TimeAverage_e01.pdf')


plt.figure(1)
plt.title('$q_g$ with waves and no waves')
plt.plot(time,np.abs(qg_wave),'k',lw=3)
plt.savefig('qg_wave.pdf')

plt.figure(2)
plt.ylabel('$q_\gamma$')
plt.plot(time,np.abs(qg_nowave),'k',lw=3)
plt.savefig('qg_nowave.pdf')

