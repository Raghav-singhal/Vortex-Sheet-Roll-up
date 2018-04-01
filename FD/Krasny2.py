# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:58:51 2015

@author: Raghav Singhal
"""
import numpy as np
from math import cos,sin,sinh,cosh,pi,log
import matplotlib.pyplot as plt
def K1(N,delta,T):
    n=float(N)
    s=np.arange(N)/n
    dt=0.1
    
    x=s+0.01*np.sin(2*pi*s)    
    y=-0.01*np.sin(2*pi*s)
         
    def f1(x,y):
        dx=np.zeros(N)
        for j in range(N):
            for k in range(N):
                if k!=j:
                   dx[j]+=sinh(2*pi*(y[j]-y[k]))/((cosh(2*pi*(y[j]-y[k]))-cos(2*pi*(x[j]-x[k]))+delta**2))
        return (-0.5/n)*dx
    
    def f2(x,y):
        dy=np.zeros(N)
        for j in range(N):
            for k in range(N):
                if k!=j:
                    dy[j]+=sin(2*pi*(x[j]-x[k]))/((cosh(2*pi*(y[j]-y[k]))-cos(2*pi*(x[j]-x[k]))+delta**2))
        return (0.5/n)*dy
    
#    def H(x,y):
#        I=0.0
#        for j in range(N):
#            for k in range(j+1,N):
#                I+=log(cosh(2*pi*(y[j]-y[k]))-cos(2*pi*(x[j]-x[k]))+delta**2)
#            return (-0.25/(pi*N**2))*I
#    
    
    
    xk1,yk1=np.ones(N),np.ones(N)
    xk2,yk2=np.ones(N),np.ones(N)
    xk3,yk3=np.ones(N),np.ones(N)
    xk4,yk3=np.ones(N),np.ones(N)
    
    Finaltime=int(T/dt)
    
    #Hamiltonian=np.ones(Finaltime)
    data=np.vstack((x,y))
    for t in range(Finaltime):
        xk1,yk1=x,y
        xk2=x + 0.5*dt*f1(xk1,yk1)
        yk2=y + 0.5*dt*f2(xk1,yk1)   
        xk3=x + 0.5*dt*f1(xk2,yk2)
        yk3=y + 0.5*dt*f2(xk2,yk2)
        xk4=x + dt*f1(xk3,yk3)
        yk4=y + dt*f2(xk3,yk3)
        x = x + (dt/6.0)*(f1(xk1,yk1) + 2*f1(xk2,yk2) + 2*f1(xk3,yk3) + f1(xk4,yk4))
        y = y + (dt/6.0)*(f2(xk1,yk1) + 2*f2(xk2,yk2) + 2*f2(xk3,yk3) + f2(xk4,yk4))
        plt.figure(t)
        plt.plot(x,y)
        data=np.hstack((data,np.vstack((x,y))))

        print (Finaltime-t)
    np.savetxt("Periodic.txt",data)
    return len(x)
"""    
    p=(x-s)+ y*1j
    
    M=int((N/2))
    phat=np.fft.fft(p)/(n)  
    phat[0]*=0.5
    
     
    def T(z): 
        I=complex(0.0,0.0)
        t=2*pi*z
        for i in range(-M,M-1):
            I+=phat[i]*(cos(i*t)+1j*sin(i*t))
        return I
    
    a=np.ones(10000,dtype=complex) 
    J=np.arange(10000)/10000.0
    for i in range(10000):
        a[i]=T(J[i]) + J[i]
     
#    return a.real,a.imag
    plt.plot(a.real,a.imag)
"""
#def plot(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11):
#    plt.figure(1)
#    plt.axis([0,1,-0.1,0.1])
#    plt.xlabel('x')
#    plt.title("N=400, $\delta=0.5$ , t=0",fontsize =20)    
#    plt.ylabel('y')
#    
#    plt.plot(x1,y1, 'k',lw=2)
#    
#    plt.savefig("K1.pdf")
#
#    
#    plt.figure(212)
#    plt.axis([0,1,-0.2,0.2])
#    plt.title("N=400, $\delta=0.5$ , t=1",fontsize =20)    
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x2,y2, 'k',lw=2)
#    plt.savefig("K2.pdf")
#
#    
#    plt.figure(3)
#    plt.axis([0,1,-0.3,0.3])
#    plt.title("N=400, $\delta=0.5$ , t=2",fontsize =20)        
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x3,y3, 'k',lw=2)
#    plt.savefig("K3.pdf")
#
#    
#    plt.figure(4)
#    plt.axis([0,1,-0.3,0.3])
#    plt.title("N=400, $\delta=0.5$ , t=3",fontsize =20)    
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x4,y4, 'k',lw=2)
#    plt.savefig("k4.pdf")
# 
#    plt.figure(5)
#    plt.axis([0,1,-0.3,0.3])
#    plt.title("N=400, $\delta=0.5$ , t=4",fontsize =20)    
#
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x5,y5, 'k',lw=2)
#    plt.savefig("k5.pdf")
#
#    plt.figure(6)
#    plt.axis([0,1,-0.1,0.1])
#    plt.title("N=400, $\delta=0.25$ , t=0",fontsize =20)    
#
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x6,y6, 'k',lw=2)
#    plt.savefig("k6.pdf")
#    
#    plt.figure(7)
#    plt.axis([0,1,-0.2,0.2])
#    plt.xlabel('x')
#    plt.title("N=400, $\delta=0.25$, t=1",fontsize =20)    
#
#    plt.ylabel('y')
#    plt.plot(x7,y7, 'k',lw=2)
#    plt.savefig("k7.pdf")
#    
#    plt.figure(8)
#    plt.axis([0,1,-0.3,0.3])
#    plt.title("N=400, $\delta=0.25$ , t=2",fontsize =20)    
#
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x8,y8, 'k',lw=2)#    
#    plt.savefig("k8.pdf")
#        
#    plt.figure(9)
#    plt.axis([0,1,-0.3,0.3])
#    plt.xlabel('x')
#    plt.title("N=400, $\delta=0.25$ , t=3",fontsize =20)    
#
#    plt.ylabel('y')
#    plt.plot(x9,y9, 'k',lw=2)
#    plt.savefig("k9.pdf")
#    
#    
#    plt.figure(10)
#    plt.axis([0,1,-0.3,0.3])
#    plt.xlabel('x')
#    plt.title("N=400, $\delta=0.25$ , t=4",fontsize =20)    
#
#    plt.ylabel('y')
#    plt.plot(x10,y10, 'k')
#    plt.savefig("k10.pdf")
#    
#    plt.figure(11)
#    plt.title("N=400, $\delta=0.05$ , t=1",fontsize =20)    
#
#    plt.axis([0.4,0.6,-0.1,0.1])
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x11,y11, 'k')
#    plt.savefig("k11.pdf")
#    
#    plt.figure(12)
#    plt.title("N=400, $\delta=0.2$ , t=1",fontsize =20)    
#
#    plt.axis([0.,1.0,-0.2,0.2])
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x11,y11, 'k')
#    plt.savefig("k12.pdf")
#    
#    plt.figure(13)
#    plt.title("N=400, $\delta=0.15$ , t=1",fontsize =20)    
#
#    plt.axis([0.0,1.,-0.2,0.2])
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x11,y11, 'k')
#    plt.savefig("k13.pdf")
#    
#    plt.figure(14)
#    plt.title("N=400, $\delta=0.1$ , t=1",fontsize =20)    
#
#    plt.axis([0.0,1.,-0.2,0.2])
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x11,y11, 'k')
#    plt.savefig("k14.pdf")
#    
#    
#    
#x1,y1=K1(400,0.5,0,0.1)
#x2,y2=K1(400,0.5,1,0.1)
#x3,y3=K1(400,0.5,2,0.1)
#x4,y4=K1(400,0.5,3,0.1)
#x5,y5=K1(400,0.5,4,0.1)
#x6,y6=K1(400,0.25,0,0.05)
#x7,y7=K1(400,0.25,1,0.05)
#x8,y8=K1(400,0.25,2,0.05)
#x9,y9=K1(400,0.25,3,0.05)
#x10,y10=K1(400,0.25,4,0.05)
#x11,y11=K1(400,0.05,1,0.01)
#x12,y12=K1(400,0.2,1,0.05)
#x13,y13=K1(400,0.15,1,0.05)
#x14,y14=K1(400,0.1,1,0.05)
#
#
#
#
#plot(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11)
