# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:17:54 2015

@author: raghavsinghal
"""

#from numba import jit,float64
from scipy import interpolate
import numpy as np
from math import cos,pi,sin,sqrt,log,cosh,sinh
import matplotlib.pyplot as plt

def pvt(N,alpha,T):
    N=int(N)
    z=np.arange(N)/(float(N))    

    x=z + 0.1*np.sin(2*pi*z)    
    y=-0.1*np.sin(2*pi*z)
    b=0.05*np.ones(N)
    
    delta=0
    #alpha=10
    etol=0.9*min(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2))
    
    x,y,z,b=RK4(x,y,z,b,T,delta,etol,alpha)        
      
    #xtck=interpolate.splrep(z,x,np.ones(len(x)),k=3)
    #ytck=interpolate.splrep(z,y,np.ones(len(x)),k=3)
    
    #x=interpolate.splev(z,xtck)
    #y=interpolate.splev(z,ytck)
    x1,y1,x2,y2=curve(x,y,z,b,1)
    
    
    plt.figure(2)
    #plt.axis([0,1,-0.3,0.3])
    #plt.title("N=400, b=0.1, |A|=0.2, t=1.2, interval for line integral=5*b")
    plt.plot(x,y,'r')   
    plt.plot(x2,y2,'k')    
    plt.plot(x1,y1,'k')
    #plt.savefig('t12_B01.pdf')
        
def weight(z1):  #quadrature weights
    return np.ones(len(z1))

def thickness(l,x,y,b):
    N=len(x)
    lnew=np.ones(N)
    for i in range(1,N-1):
         lnew[i]=sqrt(((x[i-1] - x[i+1])*0.5)**2 + ((y[i-1] - y[i+1])*0.5)**2)
    lnew[0]=sqrt(((x[0] + x[1])*0.5 - x[0])**2 + ((y[0] + y[1])*0.5 - y[0])**2)
    lnew[N-1]=sqrt(((x[N-1] + x[N-2])*0.5 - x[N-1])**2 + ((y[N-1] + y[N-2])*0.5 - y[N-1])**2)
       
    b=b*l/lnew

    return b
    
#@jit(float64(float64[:],float64[:],float64[:],float64[:],float64,float64,float64))
def RK4(x,y,z,b,T,delta,etol,alpha):
    dt=0.1
    Finaltime=int(T/dt)
    w=weight(z)
    
    xdata,ydata=x,y 
    x1data,y1data,x2data,y2data=curve(x,y,z,b,1)
    Ndata=np.array([len(x)])

    for t in range(Finaltime):        
        N=len(x)
        l=np.ones(N)
        for i in range(1,N-1):
            l[i]=sqrt(((x[i-1] - x[i+1])*0.5)**2 + ((y[i-1] - y[i+1])*0.5)**2)
        l[0]=sqrt(((x[0] + x[1])*0.5 - x[0])**2 + ((y[0] + y[1])*0.5 - y[0])**2)
        l[N-1]=sqrt(((x[N-1] + x[N-2])*0.5 - x[N-1])**2 + ((y[N-1] + y[N-2])*0.5 - y[N-1])**2)
        
        xk1,yk1=x,y

        F2=F(xk1,yk1,w,z,b,delta,alpha)
        xk2=x +  0.5*dt*F2[0]
        yk2=y +  0.5*dt*F2[1]
      
        F3=F(xk2,yk2,w,z,b,delta,alpha)
        xk3=x + 0.5*dt*F3[0]
        yk3=y + 0.5*dt*F3[1]
      
        F4=F(xk3,yk3,w,z,b,delta,alpha)
        xk4=x + dt*F4[0]
        yk4=y + dt*F4[1]

        F5=F(xk4,yk4,w,z,b,delta,alpha)
        x= x + (dt/6.)*(F2[0] + 2*F3[0] + 2*F4[0] + F5[0])
        y= y + (dt/6.)*(F2[1] + 2*F3[1] + 2*F4[1] + F5[1])
                  
        b=thickness(l,x,y,b)        
        #x,y,z,b=pointinsertion1(x,y,z,b)
        x,y,z,b=pointremoval(x,y,z,b,etol)

        
        xtck=interpolate.splrep(z,x,np.ones(len(x)),k=3)
        ytck=interpolate.splrep(z,y,np.ones(len(x)),k=3)
    
        x=interpolate.splev(z,xtck)
        y=interpolate.splev(z,ytck)
        
        x1,y1,x2,y2=curve(x,y,z,b,1)
       
        x1data=np.append(x1data,x1)   
        y1data=np.append(y1data,y1)

        x2data=np.append(x2data,x2)
        y2data=np.append(y2data,y2)

        xdata=np.append(xdata,x)        
        ydata=np.append(ydata,y)    
        
        Ndata=np.append(Ndata,len(x))
                   
#        xhat=np.fft.fft(x-z)
#        yhat=np.fft.fft(y)
#        
#        for i in range(len(xhat)):
#            if abs(xhat[i])<1e-3:
#                xhat[i]=0
#            if abs(yhat[i])<1e-3:
#                yhat[i]=0
#
#        x=(np.fft.ifft(xhat) + z).real
#        y=(np.fft.ifft(yhat)).real
                

      
        if (Finaltime-t)%5==0:
           print (Finaltime-t),max(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2)),min(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2)),len(x)

    np.savetxt("pvtxdata.txt",xdata)
    np.savetxt("pvtydata.txt",ydata)
    np.savetxt("pvtx1data.txt",x1data)
    np.savetxt("pvty1data.txt",y1data)
    np.savetxt("pvtx2data.txt",x2data)
    np.savetxt("pvty2data.txt",y2data)
    np.savetxt("pvtNdata.txt",Ndata)

                

    return x,y,z,b
    
#@jit(float64(float64[:],float64[:],float64[:],float64[:],float64))
def F(x,y,w,z,b,delta,alpha):
    N=len(x)
    n=float(N)

    l=np.ones(N)
    for i in range(1,N-1):
        l[i]=sqrt(((x[i-1] - x[i+1])*0.5)**2 + ((y[i-1] - y[i+1])*0.5)**2)
    l[0]=sqrt(((x[0] + x[1])*0.5 - x[0])**2 + ((y[0] + y[1])*0.5 - y[0])**2)
    l[N-1]=sqrt(((x[N-1] + x[N-2])*0.5 - x[N-1])**2 + ((y[N-1] + y[N-2])*0.5 - y[N-1])**2)
    

    
    xdown,ydown,xup,yup=curve(x,y,z,b,0)  
    xside1,yside1,xside2,yside2=curve(x,y,z,b*0.5,0)
    xside3,yside3,xside4,yside4=curve(x,y,z,b*0.25,0)    
    xside5,yside5,xside6,yside6=curve(x,y,z,b*0.75,0)    
        
    x=np.append(x-1,np.append(x,x+1))
    y=np.append(y,np.append(y,y))
    
    xdown=np.append(xdown-1,np.append(xdown,xdown+1))
    ydown=np.append(ydown,np.append(ydown,ydown))
    
    xup=np.append(xup-1,np.append(xup,xup+1))
    yup=np.append(yup,np.append(yup,yup))
        
    xside1=np.append(xside1-1,np.append(xside1,xside1+1))
    yside1=np.append(yside1,np.append(yside1,yside1))    

    xside2=np.append(xside2-1,np.append(xside2,xside2+1))
    yside2=np.append(yside2,np.append(yside2,yside2))    
    
    xside3=np.append(xside3-1,np.append(xside3,xside3+1))
    yside3=np.append(yside3,np.append(yside3,yside3))    

    xside4=np.append(xside4-1,np.append(xside4,xside4+1))
    yside4=np.append(yside4,np.append(yside4,yside4))    

    xside5=np.append(xside5-1,np.append(xside5,xside5+1))
    yside5=np.append(yside5,np.append(yside5,yside5))        
    
    xside6=np.append(xside6-1,np.append(xside6,xside6+1))
    yside6=np.append(yside6,np.append(yside6,yside6))    
    
    dx=np.zeros(N)
    dy=np.zeros(N)         
    
    #xmid=np.append(x[0],np.append((x[0:N-1] + x[1:N])*0.5,x[N-1]))
    #ymid=np.append(y[0],np.append((y[0:N-1] + y[1:N])*0.5,y[N-1]))   
            
    for j in range(N,2*N):
        t=1        
        length=0
        while length<b[j-N]*alpha :
              for a in range(t):
                  length+=sqrt((x[j]-x[j+a])**2 + (y[j] - y[j+a])**2)
        
              t=t+1
        
        s=1
        length=0
        while length<b[j-N]*alpha:
             for a in range(s):
                 length+=sqrt((x[j]-x[j-a])**2 + (y[j] - y[j-a])**2)        
             s=s+1

        for k in range(j-s-N):
            num1=-b[k]*l[k]*sinh(2*pi*(y[j] - y[k]))
            num2=b[k]*l[k]*sin(2*pi*(x[j] - x[k]))
            denom=(cosh(2*pi*(y[j] - y[k])) - cos(2*pi*(x[j]-x[k])))*n
            dx[j-N]+=num1/denom
            dy[j-N]+=num2/denom

        for k in range(j+t-N,N):
            num1=-b[k]*l[k]*sinh(2*pi*(y[j] - y[k]))
            num2=b[k]*l[k]*sin(2*pi*(x[j] - x[k]))
            denom=(cosh(2*pi*(y[j] - y[k])) - cos(2*pi*(x[j]-x[k])))*n
            dx[j-N]+=num1/denom
            dy[j-N]+=num2/denom
             
        xboundary1=np.array([xside5[j-s],xside1[j-s],xside3[j-s],x[j-s],xside4[j-s],xside2[j-s],xside6[j-s]])
        xboundary2=np.append(xboundary1,xup[j-s:j+t+1])
        xboundary3=np.append(xboundary2,np.array([xside6[j+t],xside2[j+t],xside4[j+t],x[j+t],xside3[j+t],xside1[j+t],xside5[j+t]]))   
            
        yboundary1=np.array([yside5[j-s],yside1[j-s],yside3[j-s],y[j-s],yside4[j-s],yside2[j-s],yside6[j-s]])
        yboundary2=np.append(yboundary1,yup[j-s:j+t+1])
        yboundary3=np.append(yboundary2,np.array([yside6[j+t],yside2[j+t],yside4[j+t],y[j+t],yside3[j+t],yside1[j+t],yside5[j+t]]))           
                        
        xboundary=np.append(xboundary3,np.flipud(xdown[j-s:j+t+1]))
        yboundary=np.append(yboundary3,np.flipud(ydown[j-s:j+t+1]))
            
        M=len(xboundary)
        r=np.zeros(M)
        for m in range(M):
            r[m]=sqrt((x[j] - xboundary[m])**2 + (y[j]- yboundary[m])**2)
            
        h=np.zeros(M)
        h[0:M-1]=np.sqrt((xboundary[1:M] - xboundary[0:M-1])**2 + (yboundary[1:M] - yboundary[0:M-1])**2)
        h[M-1]=sqrt((xboundary[M-1] - xboundary[0])**2 + (yboundary[M-1] - yboundary[0])**2)
            
        theta=np.zeros(M)
        theta[0:M-1]=np.arctan2((yboundary[1:M] - yboundary[0:M-1]),(xboundary[1:M] - xboundary[0:M-1]))
        theta[M-1]=np.arctan2((yboundary[0] - yboundary[M-1]),(xboundary[0] - xboundary[M-1]))
            
        for m in range(M):
            dx[j-N]+=h[m]*log(r[m])*cos(theta[m])/pi
            dy[j-N]+=h[m]*log(r[m])*sin(theta[m])/pi
            
    

    return np.vstack((0.5*dx,0.5*dy))

    

#@jit(float64(float64[:],float64[:],float64[:],float64[:],float64))
def curve(x,y,z,b,integral):
    N=len(x)
    slope=np.zeros(N)
    for i in range(3,N-1):
        A=np.array([[z[i-3]**4,z[i-3]**3,z[i-3]**2,z[i-3],1.],[z[i-2]**4,z[i-2]**3,z[i-2]**2,z[i-2],1.],[z[i-1]**4,z[i-1]**3,z[i-1]**2,z[i-1],1.],[z[i]**4,z[i]**3,z[i]**2,z[i],1.],[z[i+1]**4,z[i+1]**3,z[i+1]**2,z[i+1],1.]])
        u=np.array([x[i-3],x[i-2],x[i-1],x[i],x[i+1]])
        v=np.array([y[i-3],y[i-2],y[i-1],y[i],y[i+1]])
        
        a=np.linalg.solve(A,u)
        c=np.linalg.solve(A,v)
        
        slope[i-3]=(4*c[0]*z[i-3]**3 + 3*c[1]*z[i-3]**2 +2*c[2]*z[i-3] + c[3])/(4*a[0]*z[i-3]**3 + 3*a[1]*z[i-3]**2 +2*a[2]*z[i-3] + a[3])
        slope[i-2]=(4*c[0]*z[i-2]**3 + 3*c[1]*z[i-2]**2 +2*c[2]*z[i-2] + c[3])/(4*a[0]*z[i-2]**3 + 3*a[1]*z[i-2]**2 +2*a[2]*z[i-2] + a[3])
        slope[i-1]=(4*c[0]*z[i-1]**3 + 3*c[1]*z[i-1]**2 +2*c[2]*z[i-1] + c[3])/(4*a[0]*z[i-1]**3 + 3*a[1]*z[i-1]**2 +2*a[2]*z[i-1] + a[3])
        slope[i]=(4*c[0]*z[i]**3 + 3*c[1]*z[i]**2 +2*c[2]*z[i] + c[3])/(4*a[0]*z[i]**3 + 3*a[1]*z[i]**2 +2*a[2]*z[i] + a[3])
        slope[i+1]=(4*c[0]*z[i+1]**3 + 3*c[1]*z[i+1]**2 +2*c[2]*z[i+1] + c[3])/(4*a[0]*z[i+1]**3 + 3*a[1]*z[i+1]**2 +2*a[2]*z[i+1] + a[3])
    
    xtck=interpolate.splrep(z,x,np.ones(len(x)),k=3)
    ytck=interpolate.splrep(z,y,np.ones(len(x)),k=3)
    
    x=interpolate.splev(z,xtck)
    y=interpolate.splev(z,ytck)   
    
    
    
    if integral==1:
        xmid=np.append(x[0],np.append((x[0:N-1] + x[1:N])*0.5,x[N-1]))
        ymid=np.append(y[0],np.append((y[0:N-1] + y[1:N])*0.5,y[N-1]))   
    else:
        xmid,ymid=x,y
        
     
    
    x1=np.zeros(N)
    y1=np.zeros(N)
    x2=np.zeros(N)
    y2=np.zeros(N)
    
    for i in range(N):                        
        if z[i]<=0:
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
                x1[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
                y2[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
                x2[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        
            else:
                y1[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
                x1[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
                y2[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
                x2[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        if z[i]>0:
            if (xmid[i]-xmid[i-1])<=0 and (ymid[i]-ymid[i-1])>0:
                y1[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
                x1[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
                y2[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
                x2[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        
            elif (xmid[i]-xmid[i-1])<=0 and (ymid[i]-ymid[i-1])<=0:
                y1[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
                x1[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
                y2[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
                x2[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        
            elif (xmid[i]-xmid[i-1])>=0 and (ymid[i]-ymid[i-1])<0:
                y1[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
                x1[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
                y2[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
                x2[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
        
            else:
                y1[i]=ymid[i] - b[i]*0.5/sqrt(1+slope[i]**2)
                x1[i]=xmid[i] + b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
            
                y2[i]=ymid[i] + b[i]*0.5/sqrt(1+slope[i]**2)
                x2[i]=xmid[i] - b[i]*0.5*sqrt((slope[i]**2)/(1. + slope[i]**2))
    

    return x1,y1,x2,y2

def T(z,phat,N): 
    M=int(N*0.5)
    I=complex(0.0,0.0)
    t=2*pi*z
    for i in range(-M,M-1):
        I+=phat[i]*(cos(i*t)+1j*sin(i*t))
    return I
    
def interpolate1(x,y,s):    
    N=len(x)
    n=float(N)
    p=(x-s)+ y*1j
    
    phat=np.fft.fft(p)/(n)  
    phat[0]*=0.5
     
    a=np.ones(N,dtype=complex) 
    J=np.arange(N)/float(N-1.)
    for i in range(N):
        a[i]=T(J[i],phat,N) + J[i]
     
    return a.real,a.imag
    
def pointremoval(x,y,z,b,etol):
    N=len(x)
    
    X,Y,Z,B=x,y,z,b
    
    for i in range(N-1):
        if sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)<etol:
            
            Xleft=X[0:i]
            Xright=X[i+1:]
            X=np.append(Xleft,Xright)
            
            Yleft=Y[0:i]
            Yright=Y[i+1:]
            Y=np.append(Yleft,Yright)
 
            Zleft=Z[0:i]
            Zright=Z[i+1:]
            Z=np.append(Zleft,Zright)
 
            Bleft=B[0:i]
            Bright=B[i+1:]
            B=np.append(Bleft,Bright)
            
            
    return X,Y,Z,B
             
def pointinsertion1(x,y,z,b):    
    #N=len(x)-2
    N=len(x)
    etol=0.02
    for i in range(2,N-1):
        if sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)>etol:
        
        
           M=np.ones((4,4))
           for j in range(4):
               for k in range(4):
                   M[j,k]=z[i+1-j]**k
        
           if np.linalg.det(M)==0:
               break
                
           u=(np.array([x[i+1],x[i],x[i-1],x[i-2]]))
           v=(np.array([y[i+1],y[i],y[i-1],y[i-2]]))
           w=(np.array([b[i+1],b[i],b[i-1],b[i-2]]))

        
           a=np.linalg.solve(M,u)
           c=np.linalg.solve(M,v)
           d=np.linalg.solve(M,w)
        
           znew=(z[i-1] + z[i])*0.5
        
           xnew=a[3]*znew**3 + a[2]*znew**2 + a[1]*znew + a[0]
           ynew=c[3]*znew**3 + c[2]*znew**2 + c[1]*znew + c[0]
           bnew=d[3]*znew**3 + d[2]*znew**2 + d[1]*znew + d[0]
        
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
        
           b1=b[:i]
           b2=b[i:]
           b1=np.append(b1,bnew)
           b=np.append(b1,b2)
        

 
        
    print len(x),'total points'
    return x,y,z,b
        
    
#def F1(x,y,w,b,delta):
#    N=len(x)
#    n=float(N)
#    z=np.linspace(-1,1,N)
#
#    l=np.ones(N)
#    for i in range(1,N-1):
#        l[i]=sqrt(((x[i-1] - x[i+1])*0.5)**2 + ((y[i-1] - y[i+1])*0.5)**2)
#    l[0]=sqrt(((x[0] + x[1])*0.5 - x[0])**2 + ((y[0] + y[1])*0.5 - y[0])**2)
#    l[N-1]=sqrt(((x[N-1] + x[N-2])*0.5 - x[N-1])**2 + ((y[N-1] + y[N-2])*0.5 - y[N-1])**2)
#    
#    xdown,ydown,xup,yup=curve(x,y,z,b)  
#    xside1,yside1,xside2,yside2=curve(x,y,z,b*0.5)
#    xside3,yside3,xside4,yside4=curve(x,y,z,b*0.25)    
#    xside5,yside5,xside6,yside6=curve(x,y,z,b*0.75)    
#
#    dx=np.zeros(N)
#    dy=np.zeros(N)         
#    
#    dl=int(delta)
#    
#    for j in range(dl,N-dl):
#        for k in range(0,j-dl):
#            num1=-b[k]*l[k]*sinh(2*pi*(y[j] - y[k]))
#            num2=b[k]*l[k]*sin(2*pi*(x[j] - x[k]))
#            denom=(cosh(2*pi*(y[j] - y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2)*n
#            dx[j]+=num1/denom
#            dy[j]+=num2/denom
#
#        for k in range(j+dl,N):
#            num1=-b[k]*l[k]*sinh(2*pi*(y[j] - y[k]))
#            num2=b[k]*l[k]*sin(2*pi*(x[j] - x[k]))
#            denom=(cosh(2*pi*(y[j] - y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2)*n
#            dx[j]+=num1/denom
#            dy[j]+=num2/denom
#             
#        xboundary1=np.array([xside5[j-dl],xside1[j-dl],xside3[j-dl],x[j-dl],xside4[j-dl],xside2[j-dl],xside6[j-dl]])
#        xboundary2=np.append(xboundary1,xup[j-dl:j+dl+1])
#        xboundary3=np.append(xboundary2,np.array([xside6[j+dl],xside2[j+dl],xside4[j+dl],x[j+dl],xside3[j+dl],xside1[j+dl],xside5[j+dl]]))   
#            
#        yboundary1=np.array([yside5[j-dl],yside1[j-dl],yside3[j-dl],y[j-dl],yside4[j-dl],yside2[j-dl],yside6[j-dl]])
#        yboundary2=np.append(yboundary1,yup[j-dl:j+dl+1])
#        yboundary3=np.append(yboundary2,np.array([yside6[j+dl],yside2[j+dl],yside4[j+dl],y[j+dl],yside3[j+dl],yside1[j+dl],yside5[j+dl]]))           
#                        
#        xboundary=np.append(xboundary3,np.flipud(xdown[j-dl:j+dl+1]))
#        yboundary=np.append(yboundary3,np.flipud(ydown[j-dl:j+dl+1]))
#            
#        M=len(xboundary)
#        r=np.zeros(M)
#        for m in range(M):
#            r[m]=sqrt((x[j] - xboundary[m])**2 + (y[j]- yboundary[m])**2)
#            
#        h=np.zeros(M)
#        h[0:M-1]=np.sqrt((xboundary[1:M] - xboundary[0:M-1])**2 + (yboundary[1:M] - yboundary[0:M-1])**2)
#        h[M-1]=sqrt((xboundary[M-1] - xboundary[0])**2 + (yboundary[M-1] - yboundary[0])**2)
#            
#        theta=np.zeros(M)
#        theta[0:M-1]=np.arctan2((yboundary[1:M] - yboundary[0:M-1]),(xboundary[1:M] - xboundary[0:M-1]))
#        theta[M-1]=np.arctan2((yboundary[0] - yboundary[M-1]),(xboundary[0] - xboundary[M-1]))
#            
#        for m in range(M):
#            dx[j]+=h[m]*log(r[m])*cos(theta[m])/pi
#            dy[j]+=h[m]*log(r[m])*sin(theta[m])/pi
#            
#    
#    for j in range(dl):
#        
#        for k in range(j+dl,N):
#            num1=-b[k]*l[k]*sinh(2*pi*(y[j] - y[k]))
#            num2=b[k]*l[k]*sin(2*pi*(x[j] - x[k]))
#            denom=(cosh(2*pi*(y[j] - y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2)*n
#            dx[j]+=num1/denom
#            dy[j]+=num2/denom
#             
#        xboundary1=np.array([xside5[N-1-(dl-j)]-1,xside1[N-1-(dl-j)]-1,xside3[N-1-(dl-j)]-1,x[N-1-(dl-j)]-1,xside4[N-1-(dl-j)]-1,xside2[N-1-(dl-j)]-1,xside6[N-1-(dl-j)]-1])
#        xboundary2=np.append(xboundary1,np.append(xup[N-1-(dl-j):]-1,xup[:j+dl+1]))
#        xboundary3=np.append(xboundary2,np.array([xside6[j+dl],xside2[j+dl],xside4[j+dl],x[j+dl],xside3[j+dl],xside1[j+dl],xside5[j+dl]]))   
#
#        yboundary1=np.array([yside5[N-1-(dl-j)],yside1[N-1-(dl-j)],yside3[N-1-(dl-j)],y[N-1-(dl-j)],yside4[N-1-(dl-j)],yside2[N-1-(dl-j)],yside6[N-1-(dl-j)]])
#        yboundary2=np.append(yboundary1,np.append(yup[N-1-(dl-j):],yup[:j+dl+1]))
#        yboundary3=np.append(yboundary2,np.array([yside6[j+dl],yside2[j+dl],yside4[j+dl],y[j+dl],yside3[j+dl],yside1[j+dl],yside5[j+dl]]))           
#                        
#        xboundary=np.append(xboundary3,np.flipud(np.append(xdown[N-1-(dl-j):]-1,xdown[:j+dl+1])))
#        yboundary=np.append(yboundary3,np.flipud(np.append(ydown[N-1-(dl-j):],ydown[:j+dl+1])))
#        
#        M=len(xboundary)
#        r=np.zeros(M)
#        for m in range(M):
#            r[m]=sqrt((x[j] - xboundary[m])**2 + (y[j]- yboundary[m])**2)
#            
#        h=np.zeros(M)
#        h[0:M-1]=np.sqrt((xboundary[1:M] - xboundary[0:M-1])**2 + (yboundary[1:M] - yboundary[0:M-1])**2)
#        h[M-1]=sqrt((xboundary[M-1] - xboundary[0])**2 + (yboundary[M-1] - yboundary[0])**2)
#            
#        theta=np.zeros(M)
#        theta[0:M-1]=np.arctan2((yboundary[1:M] - yboundary[0:M-1]),(xboundary[1:M] - xboundary[0:M-1]))
#        theta[M-1]=np.arctan2((yboundary[0] - yboundary[M-1]),(xboundary[0] - xboundary[M-1]))
#            
#        for m in range(M):
#            dx[j]+=h[m]*log(r[m])*cos(theta[m])/pi
#            dy[j]+=h[m]*log(r[m])*sin(theta[m])/pi 
#
#   
#    for j in range(N-dl,N):
#        for k in range(0,j-dl):
#            num1=-b[k]*l[k]*sinh(2*pi*(y[j] - y[k]))
#            num2=b[k]*l[k]*sin(2*pi*(x[j] - x[k]))
#            denom=(cosh(2*pi*(y[j] - y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2)*n
#            dx[j]+=num1/denom
#            dy[j]+=num2/denom
#
#        xboundary1=np.array([xside5[j-dl],xside1[j-dl],xside3[j-dl],x[j-dl],xside4[j-dl],xside2[j-dl],xside6[j-dl]])
#        xboundary2=np.append(xboundary1,np.append(xup[j-dl:],xup[:dl+N-j+1])+1)            
#        xboundary3=np.array([xside5[dl+N-j]+1,xside1[dl+N-j]+1,xside3[dl+N-j]+1,x[dl+N-j]+1,xside4[dl+N-j]+1,xside2[dl+N-j]+1,xside6[dl+N-j]+1])
#            
#        yboundary1=np.array([yside5[j-dl],yside1[j-dl],yside3[j-dl],y[j-dl],yside4[j-dl],yside2[j-dl],yside6[j-dl]])
#        yboundary2=np.append(yboundary1,np.append(yup[j-dl:],yup[:dl+N-j+1]))
#        yboundary3=np.array([yside5[dl+N-j],yside1[dl+N-j],yside3[dl+N-j],y[dl+N-j],yside4[dl+N-j],yside2[dl+N-j],yside6[dl+N-j]])
#                        
#        xboundary=np.append(xboundary3,np.flipud(np.append(xdown[j-dl:],xdown[:1+dl+N-j]+1)))
#        yboundary=np.append(yboundary3,np.flipud(np.append(ydown[j-dl:],ydown[:1+dl+N-j])))
#            
#        M=len(xboundary)
#        r=np.zeros(M)
#        for m in range(M):
#            r[m]=sqrt((x[j] - xboundary[m])**2 + (y[j]- yboundary[m])**2)
#            
#        h=np.zeros(M)
#        h[0:M-1]=np.sqrt((xboundary[1:M] - xboundary[0:M-1])**2 + (yboundary[1:M] - yboundary[0:M-1])**2)
#        h[M-1]=sqrt((xboundary[M-1] - xboundary[0])**2 + (yboundary[M-1] - yboundary[0])**2)
#            
#        theta=np.zeros(M)
#        theta[0:M-1]=np.arctan2((yboundary[1:M] - yboundary[0:M-1]),(xboundary[1:M] - xboundary[0:M-1]))
#        theta[M-1]=np.arctan2((yboundary[0] - yboundary[M-1]),(xboundary[0] - xboundary[M-1]))
#            
#        for m in range(M):
#            dx[j]+=h[m]*log(r[m])*cos(theta[m])/pi
#            dy[j]+=h[m]*log(r[m])*sin(theta[m])/pi
#                    
#
##    plt.figure(2)
##    plt.plot(np.flipud(np.append(xup[N-1-(dl-4):]-1,xup[:4+dl+1])),np.flipud(np.append(yup[N-1-(dl-4):],yup[:4+dl+1])),'k')
##    plt.plot(np.flipud(np.append(xdown[N-1-(dl-4):]-1,xdown[:4+dl+1])),np.flipud(np.append(ydown[N-1-(dl-4):],ydown[:4+dl+1])),'k')
#
##    plt.figure(3)
#                        
##    plt.plot(np.flipud(np.append(xdown[195-dl:],xdown[:1+dl+N-195]+1)),np.flipud(np.append(ydown[195-dl:],ydown[:1+dl+N-195])),'ko')
#    
#    
#
#
#    return np.vstack((0.5*dx,0.5*dy))

