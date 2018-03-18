# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:28:33 2015

@author: raghavsinghal
"""
from mpmath import *
import numpy as np
from math import log,cos,pi,sqrt,sin
import matplotlib.pyplot as plt
#from numba import jit,float64

#@jit(float64(float64,float64))
def ellipse(N,T):
    N=int(N)
    n=float(N)
    
    z=np.linspace(0,2*pi,N)    
    x=0.2*np.cos(z)
    y=0.1*np.sin(z)
    
    x,y=RK4(x,y,z,T)
    #X=FRK(1,x,y,z,T)
    #plt.scatter(X[0],X[1])
    plt.plot(x,y)    
    #print z

#def f(Arb,X):
def f(x,y):

    #x=X[0]
    #y=X[1]
    N=len(x)
    R=2
    
    r=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j!=i:
                r[i,j]=sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

    h=np.zeros(N)
    for i in range(N-1):
        h[i]=sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
    h[N-1]=sqrt((x[N-1] - x[0])**2 + (y[N-1] - y[0])**2)
    
    theta=np.zeros(N)
    for i in range(N-1):
        theta[i]=atan2(y[i+1] - y[i],x[i+1] - x[i])
    theta[N-1]=atan2(y[0] - y[N-1],x[0] - x[N-1])
    
    phi=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j!=i:
                phi[i,j]=atan2((y[i] - y[j]),(x[i] - x[j]))
    
    u=np.zeros((N,N))    
    for i in range(N):
        for j in range(N):
            if j!=i:
                u[i,j]=cos(phi[i,j] - theta[j])
    
    B=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j!=i:
                B[i,j]=(h[j]/r[i,j])*(sqrt(1 - u[i,j]**2))/(1 - h[j]*u[i,j]/r[i,j])

    q=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            q[i,j]=1.0 - 2*(h[j]*u[i,j]/r[i,j]) + (h[j]/r[i,j])**2
            
    Du=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j!=i-1 and j!=i:# and j!=i+1:
                Du[i,j]=(1.0/(2*pi))*h[j]*log(r[i,j]/R)
                #Du[i,j]=(1./(2*pi))*h[j]*(log(r[i,j]/R) - 0.5*u[i,j]*h[j]/r[i,j] + (1./6.)*(1 - 2*u[i,j]**2)*(h[j]/r[i,j])**2 + 0.25*u[i,j]*(1 - (4./3.)*u[i,j]**2)*(h[j]/r[i,j])**3)
                #Du[i,j]+=(1.0/(4*pi))*h[j]*(log(r[i,j]/R)**2 + (1.0 - u[i,j]*r[i,j]/h[j])*log(q[i,j]) - 2.0 + 2.*(sqrt(1.0 - u[i,j]**2))*(r[i,j]/h[j])*np.arctan(B[i,j]))
            elif j==i and j==i-1:
                Du[i,j]=(1.0/(2*pi))*h[j]*(log(h[j]/R)-1.0)
        
        #Du[i,0]=(1.0/(2*pi))*h[0]*(log(h[0]/R) - 1.)
        #Du[i,N-1]=(1.0/(2*pi))*h[N-1]*(log(h[N-1]/R) -1.)

    dx,dy=np.zeros(N),np.zeros(N)
    for i in range(N):
        for j in range(N):
            dx[i]+=Du[i,j]*cos(theta[j])
            dy[i]+=Du[i,j]*sin(theta[j])
    return np.vstack((dx,dy))
    
def RK4(x,y,z,T):
    dt=0.2
    finaltime=int(T/dt)
    
    for t in range(finaltime):
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
        #x,y,z=pointinsertion(x,y,z)
        
        #x,y,z=pointremoval(x,y,z)
        
        print T- dt*t

    return x,y
    
#@jit(float64(float64,float64[:],float64[:],float64))
def FRK(e,x,y,z,T):
    e=1e-1
    dx=0.2
    #xdata,ydata=x,y
    
    y=np.vstack((x,y))
    t=np.zeros(1)

    k1=0.0
    k2=0.0
    k3=0.0
    k4=0.0
    k5=0.0
    k6=0.0
    n=0.0
    for x in t:
        while x<=T and n*dx<=T:
              k1=y

              k2=y+dx*0.25*f(x,k1)

              k3=y+dx*((3.0/32.0)*f(x,k1)+(9.0/32.0)*f(x+(0.25)*dx,k2))

              k4=y+dx*((1932.0/2197.0)*f(x,k1)-(7200.0/2197.0)*f(x+(0.25)*dx,k2)+(7296.0/2197.0)*f(x+(3.0/8.0)*dx,k3))

              k5=y+dx*((439.0/216.0)*f(x,k1)-8*f(x+(0.25)*dx,k2)+(3680.0/513.0)*f(x+(3.0/8.0)*dx,k3)-(845.0/4104.0)*f(x+(12.0/13.0)*dx,k4))

              k6=y+dx*((-8.0/27.0)*f(x,k1)+2*f(x+(0.25)*dx,k2)-(3544.0/2565.0)*f(x+(3.0/8.0)*dx,k3)+(1859.0/4104.0)*f(x+(12.0/13.0)*dx,k4)-(11.0/40.0)*f(x+dx,k5))

              error=dx*abs((25.0/216.0)*f(x,k1)+(0)+(1408.0/2565.0)*f(x+(3.0/8.0)*dx,k3)+(2197.0/4104.0)*f(x+(12.0/13.0)*dx,k4)-0.2*f(x+dx,k5)-(16.0/135.0)*f(x,k1)+(0)+(6656.0/12825.0)*f(x+(3.0/8.0)*dx,k3)+(28561/56430.0)*f(x+(12.0/13.0)*dx,k4)-(9.0/50.0)*f(x+dx,k5)+(2.0/55.0)*f(x+0.5*dx,k6))
              
              if np.max(np.abs(error))<e or dx<1e-3 :
                 y=y+dx*((25.0/216.0)*f(x,k1)+(0)+(1408.0/2565.0)*f(x+(3.0/8.0)*dx,k3)+(2197.0/4104.0)*f(x+(12.0/13.0)*dx,k4)-0.2*f(x+dx,k5))
                 #yhat=y+dx*((16.0/135.0)*f(x,k1)+(0)+(6656.0/12825.0)*f(x+(3.0/8.0)*dx,k3)+(28561/56430.0)*f(x+(12.0/13.0)*dx,k4)-(9.0/50.0)*f(x+dx,k5)+(2.0/55.0)*f(x+0.5*dx,k6))

                 x=x+dx
                 t=np.append(t,x)
                 n+=1
                 print dx,T-x
                 
                 #xdata=np.append(xdata,y[0])
                 #ydata=np.append(ydata,y[1])
              
              else:
                 dx=0.5*dx*((e/np.max(np.abs(error)))**(0.25))
                 print dx
              x,y,z=pointinsertion(x,y,z)
    #np.savetxt('xdataADAP.txt',xdata)
    #np.savetxt('ydataADAP.txt',ydata)
    return y
#@jit(float64(float64[:],float64[:],float64[:]))
def pointinsertion(x,y,z):    
    N=len(x)
    distance=np.ones(N)    
    distance[0:N-1]=((x[1:N] - x[0:N-1])**2 + (y[1:N] - y[0:N-1])**2)**0.5
    distance[N-1]=sqrt((x[N-1] - x[0])**2 + (y[N-1] - y[0])**2)
    
    maxd=np.max(distance)
    
    etol=0.1
    t=0

    while maxd>etol and t<len(x)*0.5:
        maxarg=np.argmax(distance)
        i=maxarg        
        
        M=matrix(4)
        for j in range(4):
            for k in range(4):
                M[j,k]=z[i+1 - j]**k
        
        if det(M)==0:
            print maxd,t            
            break
                
        u=(matrix([x[i+1],x[i],x[i-1],x[i-2]]))
        v=(matrix([y[i+1],y[i],y[i-1],y[i-2]]))
        
        #a=np.linalg.solve(M,u)
        #b=np.linalg.solve(M,v)
        a=lu_solve(M,u)
        b=lu_solve(M,v)
        
        znew=(z[i-1] + z[i])*0.5
        #print znew
        
        xnew=a[3]*znew**3 + a[2]*znew**2 + a[1]*znew + a[0]
        ynew=b[3]*znew**3 + b[2]*znew**2 + b[1]*znew + b[0]
        
#        znew1=(z[i-1] + z[i])*0.75
#        
#        xnew1=a[3]*znew1**3 + a[2]*znew1**2 + a[1]*znew1 + a[0]
#        ynew1=b[3]*znew1**3 + b[2]*znew1**2 + b[1]*znew1 + b[0]
          
        z1=z[:i]
        z2=z[i:]
        z1=np.append(z1,znew)
        z=np.append(z1,z2)
        
        x1=x[:i]
        x2=x[i:]
        x1=np.append(x1,xnew)
        x=np.append(x1,x2)

        y1=y[:i]
        y2=y[i:]
        y1=np.append(y1,ynew)
        y=np.append(y1,y2)
        
        #x,y,z=pointremoval(x,y,z)
        
        N=len(x)
        distance=np.ones(N)    
        distance[0:N-1]=((x[1:N] - x[0:N-1])**2 + (y[1:N] - y[0:N-1])**2)**(0.5)
        distance[N-1]=sqrt((x[N-1] - x[0])**2 + (y[N-1] - y[0])**2)
    
        maxd=np.max(distance)
        maxarg=np.argmax(distance)
        t=t+1
 
        
    #print z       
    print len(x) - N,'added'
    return x,y,z   
    
def pointremoval(x,y,z):
    etol=1e-6
    N=len(x)
    for i in range(2,N-2):
        if min((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2)<etol:         
            z1=z[:i-1]
            z2=z[i:]
            z=np.append(z1,z2)
        
            x1=x[:i-1]
            x2=x[i:]
            x=np.append(x1,x2)

            y1=y[:i-1]
            y2=y[i:]
            y=np.append(y1,y2)
            
            
    print N-len(x),'removed'
    return x,y,z
    
    
                
                
  
          
        
            
            
    