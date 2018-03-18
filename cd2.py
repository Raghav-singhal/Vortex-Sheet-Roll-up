# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 01:55:16 2015

@author: raghavsinghal
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:42:38 2015

@author: raghavsinghal
"""

from math import log,pi,atan,cos,sin,sqrt
import numpy as np
import matplotlib.pyplot as plt

def cd2(N,T):
    n=float(N)

    z=np.arange(N)/(n)
    #z=np.linspace(0,1,N)

    x=2*np.cos(2*pi*z)
    y=np.sin(2*pi*z)
    #print x,y

    x,y=RK4(x,y,T)
    #x,y=leapfrog(x,y,T)

    plt.scatter(x,y)
    #print x,y
def f(x,y):
    N=len(x)
    R=2.0

    h=np.zeros(N)
    h[0:N-1]=np.sqrt((x[1:N] - x[0:N-1])**2 + (y[1:N] - y[0:N-1])**2)
    h[N-1]=sqrt((x[0] - x[N-1])**2 + (y[0] - y[N-1])**2)

    theta=np.zeros(N)
    theta[0:N-1]=np.arctan2((y[1:N] - y[0:N-1]),(x[1:N] - x[0:N-1]))
    theta[N-1]=np.arctan2((y[0] - y[N-1]),(x[0] - x[N-1]))

    r=np.zeros((N,N))
    for i in range(N):
        for j in range(1,N-1):
            if j!=i:
                r[i,j]=sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

    phi=np.zeros((N,N))
    for i in range(N):
        for j in range(1,N-1):
            if j!=i:
                phi[i,j]=np.arctan2((y[j]-y[i]),(x[j]-x[i]))

    u=np.zeros((N,N))
    for i in range(N):
        for j in range(1,N-1):
            if j!=i:
                u[i,j]=cos(phi[i,j] - theta[j])

    q=np.zeros((N,N))
    for i in range(N):
        for j in range(1,N-1):
            if j!=i:
                q[i,j]=1.0 - 2.0*(h[j]*u[i,j]/r[i,j]) + (h[j]/r[i,j])**2

    B=np.zeros((N,N))
    for i in range(N):
        for j in range(1,N-1):
            if j!=i:
                B[i,j]=(h[j]/r[i,j])*(sqrt(1.0 - u[i,j]**2))/(1.0 - h[j]*u[i,j]/r[i,j])

    Du=np.zeros((N,N))
    for i in range(N):
        for j in range(1,N-1):
            if j!=i and j!=i-1:# and j!=i+1:
                Du[i,j]=(1./(2*pi))*h[j]*(log(r[i,j]/R) - 0.5*u[i,j]*h[j]/r[i,j] + (1./6.)*(1 - 2*u[i,j]**2)*(h[j]/r[i,j])**2 + 0.25*u[i,j]*(1 - (4./3.)*u[i,j]**2)*(h[j]/r[i,j])**3)
                #Du[i,j]+=(1.0/(4*pi))*h[j]*(log(r[i,j]/R)**2 + (1.0 - u[i,j]*r[i,j]/h[j])*log(q[i,j]) - 2.0 + 2.*(sqrt(1.0 - u[i,j]**2))*(r[i,j]/h[j])*np.arctan(B[i,j]))
            elif j==i-1 and i:
                Du[i,j]=(1.0/(2*pi))*h[j]*(log(h[j]/R) - 1.)
            #elif j==i:
            #    Du[i,j]=
        Du[i,0]=(1.0/(2*pi))*h[0]*(log(h[0]/R) - 1.)
        Du[i,N-1]=(1.0/(2*pi))*h[N-1]*(log(h[N-1]/R) -1.)

    dx,dy=np.zeros(N),np.zeros(N)
    for i in range(N):
        for j in range(N):
            if j!=i:
               dx[i]+=Du[i,j]*cos(theta[j])
               dy[i]+=Du[i,j]*sin(theta[j])
    return np.vstack((dx,dy))

def RK4(x,y,T):
    dt=0.1
    Finaltime=int(T/dt)
    xdata,ydata=x,y

    for t in range(Finaltime):
        F1= f(x,y)
        xk1,yk1 = dt*F1[0],dt*F1[1]

        F2= f(x + 0.5*xk1,y + 0.5*yk1)
        xk2 = dt*F2[0]
        yk2 = dt*F2[1]

        F3=f(x + 0.5*xk2,y + 0.5*yk2)
        xk3 = dt*F3[0]
        yk3 = dt*F3[1]

        F4=f(x + xk3,y+ yk3)
        xk4 = dt*F4[0]
        yk4 = dt*F4[1]

        x = x + (1./6.)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        y = y + (1./6.)*(yk1 + 2*yk2 + 2*yk3 + yk4)

        xdata=np.append(xdata,x)
        ydata=np.append(ydata,y)
        print (Finaltime - t)*dt

    np.savetxt('xcd.txt',xdata)
    np.savetxt('ycd.txt',ydata)

    return x,y


def leapfrog(x,y,T):
    N=len(x)
    dt=0.2
    dt1=0.1
    finaltime=int(T/dt)
    xdata,ydata=x,y

    X,Y=np.zeros((3,N)),np.zeros((3,N))
    X[0],Y[0]=x,y

    F=np.zeros((2,N))
    F=f(x,y)
    X[1]=X[0] + 2*dt1*F[0]
    Y[1]=Y[0] + 2*dt1*F[1]

    t=1
    while t<=finaltime:
        F=f(X[1],Y[1])

        X[2]=X[0] + 2*dt*F[0]
        Y[2]=Y[0] + 2*dt*F[1]

        X[0],Y[0]=X[1],Y[1]
        X[1],Y[1]=X[2],Y[2]
        t+=1
        xdata=np.append(xdata,X[1])
        ydata=np.append(ydata,Y[1])
        print finaltime - t
    np.savetxt('xcd.txt',xdata)
    np.savetxt('ycd.txt',ydata)
    return X[1],Y[1]




