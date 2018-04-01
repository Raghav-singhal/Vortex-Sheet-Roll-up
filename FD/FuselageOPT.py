# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:25:50 2015

@author: Raghav Singhal
"""
#from scipy.fftpack import dct
import numpy as np
from math import cos,sin,pi,acos#,log
import matplotlib.pyplot as plt
def F1(M,delta,T):
    
    
    
    dt=0.02    
    finaltime=int(T/dt)
    e=0.02  #mesh parameter 
    
    m1=0.
    m2=0.3
    m3=sin(acos(0.7))
    A=np.array([[m1**3,m1**2,m1,1],[m2**3,m2**2,m2,1],[3*m1**2,2*m1,1,0],[3*m2**2,2*m2,1,0]])
    B=np.array([[m2**3,m2**2,m2,1],[m3**3,m3**2,m3,1],[3*m2**2,2*m2,1,0],[3*m3**2,2*m3,1,0]])
    u=np.array([1.4,2.,0.,0.])
    v=np.array([2.,sin(acos(0.7)),0.,-0.980196])
    a=np.linalg.solve(A,u)
    b=np.linalg.solve(B,v)
    
    def weight(z1):  #quadrature weights
        N1=len(z1)
        x1=-np.cos(z1)
        w1=np.zeros(N1,dtype=float)
        n1=float(N1)
        for i in range(N1):
            if 0.7<=abs(x1[i])<=1.:
                w1[i]=cos(z1[i])*pi/n1
            elif -0.7<=x1[i]<=-0.3:             
                w1[i]=-sin(z1[i])*(b[2]+2*b[1]*abs(x1[i])+3*b[0]*abs(x1[i])**2)*pi/n1
            elif -0.3<=x1[i]<0.:
                w1[i]=-sin(z1[i])*(a[2]+2*a[1]*abs(x1[i])+3*a[0]*abs(x1[i])**2)*pi/n1
            elif 0.0<=(x1[i])<=0.3:
                w1[i]=sin(z1[i])*(a[2]+2*a[1]*abs(x1[i])+3*a[0]*abs(x1[i])**2)*pi/n1
            elif 0.3<=(x1[i])<=0.7:
                w1[i]=sin(z1[i])*(b[2]+2*b[1]*abs(x1[i])+3*b[0]*abs(x1[i])**2)*pi/n1
        #for k in range(int(N1/2.)):
        #    w1[int(N1/2.)+k]=-w1[int(N1/2.)-k]
        
        w1[0]*=0.5
        w1[N1-1]*=0.5
        return w1
                        
    def f(p2,w2):
        
        x2=p2.real
        y2=p2.imag
        N2=len(x2)
        
        dx=np.zeros(N2)
        dy=np.zeros(N2)
        
        xKdelta=np.ones((N2,N2))
        yKdelta=np.ones((N2,N2))
        
        for j in range(N2):
            for k in range(j,N2):
                xKdelta[j,k]=(y2[j]-y2[k])/((y2[j]-y2[k])**2+(x2[j]-x2[k])**2+delta**2)
                yKdelta[j,k]=(x2[j]-x2[k])/((y2[j]-y2[k])**2+(x2[j]-x2[k])**2+delta**2)
                xKdelta[k,j]=-xKdelta[j,k]
                yKdelta[k,j]=-yKdelta[j,k]
        
            dx[j]=np.dot(w2,xKdelta[j,:])
            dy[j]=np.dot(w2,yKdelta[j,:])
                
        return -0.5*dx/pi + 1j*0.5*dy/pi

        
    def insert(x4,y4,z4,e):
        p=x4 + 1j*y4
        for i in range(1,len(p)-1):
            if abs(p[i]-p[i+1])<e:
                pass
            else:
                M=np.array(([z4[i-1]**3,z4[i-1]**2,z4[i-1],1.],[z4[i]**3,z4[i]**2,z4[i],1.],[z4[i+1]**3,z4[i+1]**2,z4[i+1],1.],[z4[i+2]**2,z4[i+2]**2,z4[i+2],1.]))
                u1=np.array([p[i-1],p[i],p[i+1],p[i+2]])
                d=np.linalg.solve(M,u1)
                znew=0.5*(z4[i]+z4[i+1])
                pnew=d[0]*znew**3 + d[1]*znew**2 + d[2]*znew + d[3]
                
                
                ZHALF=np.append(z4[0:i+1],znew)
                Zother=z4[i+1:len(p)]
                z4=np.append(ZHALF,Zother)
                
                PHALF=np.append(p[0:i+1],pnew)
                Pother=p[i+1:len(p)]
                p=np.append(PHALF,Pother)
        return p.real,p.imag,z4
       
    N=int(2*M)
    z=np.arange(N)*pi/(2.*M)
    x=-np.cos(z)
    y=np.zeros(N,dtype=float)

    
    for t in range(finaltime):
        x,y,z=insert(x,y,z,e)
        w=weight(z)
        k1=x + 1j*y
        k2=x + 1j*y + 0.5*dt*f(k1,w) 
        k3=x + 1j*y + 0.5*dt*f(k2,w)
        k4=x + 1j*y + dt*f(k3,w)
        x= (x + 1j*y + (dt/6.)*(f(k1,w) + 2*f(k2,w) + 2*f(k3,w) + f(k4,w))).real
        y= (x + 1j*y + (dt/6.)*(f(k1,w) + 2*f(k2,w) + 2*f(k3,w) + f(k4,w))).imag
        x,y,z=insert(x,y,z,e)
        print finaltime-t
        
#    plt.title("N=400, $\delta=0.1$, t=0.1",fontsize =20)    
#
#    plt.axis([-1.3,1.3,-0.3,0.1])
#    plt.xlabel('x')
#    plt.ylabel('y')
    plt.plot(x,-y, 'k',lw=2)
#    plt.savefig("FU1.pdf")
#    return x,-y

#def plot(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7):
#    plt.figure(1)
#    plt.title("N=400, $\delta=0.1$, t=0",fontsize =20)    
#
#    plt.axis([-1,1,-0.3,0.1])
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x1,y1, 'k',lw=2)
#    plt.savefig("FU1.pdf")
#
#    
#    plt.figure(2)
#    plt.title("N=400, $\delta=0.1$, t=0.3",fontsize =20)    
#
#    plt.axis([-1,1,-0.5,0.3])
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x2,y2, 'k',lw=2)
#    plt.savefig("FU2.pdf")
#
#    
#    plt.figure(3)
#    plt.axis([-2,2,-0.9,0.3])
#    plt.title("N=400, $\delta=0.1$, t=0.5",fontsize =20)    
#
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x3,y3, 'k',lw=2)
#    plt.savefig("FU3.pdf")
#
#    
#    plt.figure(4)
#    plt.axis([-2,2,-1.2,0.3])
#    plt.xlabel('x')
#    plt.title("$\delta=0.1$, t=1",fontsize =20)    
#
#    plt.ylabel('y')
#    plt.plot(x4,y4, 'k',lw=2)
#    plt.savefig("FU4.pdf")
# 
#    plt.figure(5)
#    plt.axis([-2,2,-1.7,0.3])
#    plt.title("$\delta=0.1$, t=2",fontsize =20)    
#
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x5,y5, 'k',lw=2)
#    plt.savefig("FU5.pdf")
#
#    plt.figure(6)
#    plt.axis([-2,2,-1.7,-0.1])
#    plt.title("$\delta=0.05$, t=3",fontsize =20)    
#
#    plt.xlabel('x')    
#    plt.ylabel('y')
#    plt.plot(x6,y6, 'k',lw=2)
#    plt.savefig("FU6.pdf")
#    
#    plt.figure(7)
#    plt.axis([-2,2,-1.9,-0.1])
#    plt.title("$\delta=0.05$, t=4",fontsize =20)    
#
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.plot(x7,y7, 'k',lw=2)
#    plt.savefig("FU7.pdf")
#    
#    
#    
#    
#x1,y1=F1(200,0.1,0,0.04)
#x2,y2=F1(200,0.1,0.3,0.04)
#x3,y3=F1(200,0.1,0.5,0.04)
#x4,y4=F1(200,0.1,1,0.04)
#x5,y5=F1(200,0.1,2,0.009)
#x6,y6=F1(200,0.1,3,0.04)
#x7,y7=F1(200,0.1,4,0.04)
#plot(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7)











"""    
    def plot(x,y):   FFT 
        Q=int(len(x)/2)
        A=dct(x)/(len(x))  
        A[0]*=0.5
    
        B=dct(y)/(len(x))
        B[0]*=0.5
    
    
        phat=A+1j*B
        def P(t):
            I=complex(0.,0.)
            for k in range(-Q+1,Q+1):
                I+=phat[k]*(cos(2*k*t))
            return I
    
    
        J=np.arange(20000)*pi/(20000.0)
        a=np.ones(20000,dtype=complex)
        for i in range(20000):
            a[i]=P(J[i])
    plt.plot(a.real,-a.imag,'b')

    def removal(x5,y5,z5,f):   point removal
        p1=x5+1j*y5
        for i in range(1,len(p1)-1):
            if abs(p1[i]-p1[i-1])>f:
                pass
            else:
                zhalf=z5[0:i+1]
                zother=z5[i+2:len(p1)]
                z5=np.append(zhalf,zother)
                
                p1half=p1[0:i+1]
                p1other=p1[i+2:len(p1)]
                p1=np.append(p1half,p1other)
        return p1.real,p1.imag,z5
    f=0.0001 removal parameter


"""    