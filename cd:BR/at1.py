# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:33:11 2015

@author: raghavsinghal
"""
from numba import jit,float64
import numpy as np
from math import cos,pi,sin,sqrt,log
import matplotlib.pyplot as plt

def vt(N,delta,T):
    N=int(N)
    x=np.linspace(-1,1,N)
    y=np.zeros(N)   
    z=x
    b=0.1*(1 - x**2)
    
    x,y,z,b=RK4(x,y,z,b,T,delta)    
    x1,y1,x2,y2=curve(x,y,z,b)
   
    plt.figure(1)
    plt.axis([-1,1,-0.2,0.2])
    plt.plot(x,y,'ro')   
    plt.plot(x2,y2,'ko')    
    plt.plot(x1,y1,'ko')
        
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
    

#@jit(float64(float64[:],float64[:],float64[:],float64,float64))
def RK4(x,y,z,b,T,delta):
    dt=0.1
    Finaltime=int(T/dt)
    w=weight(z)
    for t in range(Finaltime):        
        N=len(x)
        l=np.ones(N)
        for i in range(1,N-1):
            l[i]=sqrt(((x[i-1] - x[i+1])*0.5)**2 + ((y[i-1] - y[i+1])*0.5)**2)
        l[0]=sqrt(((x[0] + x[1])*0.5 - x[0])**2 + ((y[0] + y[1])*0.5 - y[0])**2)
        l[N-1]=sqrt(((x[N-1] + x[N-2])*0.5 - x[N-1])**2 + ((y[N-1] + y[N-2])*0.5 - y[N-1])**2)
        
        xk1,yk1=x,y

        F2=F(xk1,yk1,w,z,b,delta)
        xk2=x +  0.5*dt*F2[0]
        yk2=y +  0.5*dt*F2[1]
      
        F3=F(xk2,yk2,w,z,b,delta)
        xk3=x + 0.5*dt*F3[0]
        yk3=y + 0.5*dt*F3[1]
      
        F4=F(xk3,yk3,w,z,b,delta)
        xk4=x + dt*F4[0]
        yk4=y + dt*F4[1]
      
        F5=F(xk4,yk4,w,z,b,delta)
        x= x + (dt/6.)*(F2[0] + 2*F3[0] + 2*F4[0] + F5[0])
        y= y + (dt/6.)*(F2[1] + 2*F3[1] + 2*F4[1] + F5[1])

        b=thickness(l,x,y,b)

        #x,y,z,b=pointinsertion(x,y,z,b)
        #x,y,z,b=pointremoval(x,y,z,b)        

        
        
        print (Finaltime-t),max(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2)),min(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2))
            

    return x,y,z,b
    
#@jit(float64(float64[:],float64[:],float64[:],float64[:],float64))
def F(x,y,w,z,b,delta):
    N=len(x)
    n=float(N)
    #z=np.linspace(-1,1,N)

    l=np.ones(N)
    for i in range(1,N-1):
        l[i]=sqrt(((x[i-1] - x[i+1])*0.5)**2 + ((y[i-1] - y[i+1])*0.5)**2)
    l[0]=sqrt(((x[0] + x[1])*0.5 - x[0])**2 + ((y[0] + y[1])*0.5 - y[0])**2)
    l[N-1]=sqrt(((x[N-1] + x[N-2])*0.5 - x[N-1])**2 + ((y[N-1] + y[N-2])*0.5 - y[N-1])**2)

    #print l
    
    xdown,ydown,xup,yup=curve(x,y,z,b)  #opposite of main prog
    xside1,yside1,xside2,yside2=curve(x,y,z,b*0.5)
    xside3,yside3,xside4,yside4=curve(x,y,z,b*0.25)    
    xside5,yside5,xside6,yside6=curve(x,y,z,b*0.75)    

    dx=np.zeros(N)
    dy=np.zeros(N)         
    
    dl=int(delta)
    
    for j in range(dl,N-dl):
        for k in range(0,j-dl):
            num1=-b[k]*l[k]*(y[j] - y[k])
            num2=b[k]*l[k]*(x[j] - x[k])
            denom=((y[j] - y[k])**2 + (x[j]-x[k])**2 + delta**2)*n
            dx[j]+=num1/denom
            dy[j]+=num2/denom

        for k in range(j+dl,N):
            num1=-b[k]*l[k]*(y[j] - y[k])
            num2=b[k]*l[k]*(x[j] - x[k])
            denom=((y[j] - y[k])**2 + (x[j]-x[k])**2 + delta**2)*n
            dx[j]+=num1/denom
            dy[j]+=num2/denom
             
        for k in range(j-dl,j+dl): 
            xboundary1=np.array([xside5[j-dl],xside1[j-dl],xside3[j-dl],x[j-dl],xside4[j-dl],xside2[j-dl],xside6[j-dl]])
            xboundary2=np.append(xboundary1,xup[j-dl:j+dl])
            xboundary3=np.append(xboundary2,np.array([xside6[j+dl],xside2[j+dl],xside4[j+dl],x[j+dl],xside3[j+dl],xside1[j+dl],xside5[j+dl]]))   
            
            yboundary1=np.array([yside5[j-dl],yside1[j-dl],yside3[j-dl],y[j-dl],yside4[j-dl],yside2[j-dl],yside6[j-dl]])
            yboundary2=np.append(yboundary1,yup[j-dl:j+dl])
            yboundary3=np.append(yboundary2,np.array([yside6[j+dl],yside2[j+dl],yside4[j+dl],y[j+dl],yside3[j+dl],yside1[j+dl],yside5[j+dl]]))           
                        
            xboundary=np.append(xboundary3,np.flipud(xdown[j-dl:j+dl]))
            yboundary=np.append(yboundary3,np.flipud(ydown[j-dl:j+dl]))
            
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
                dx[j]+=h[m]*log(r[m])*cos(theta[m])
                dy[j]+=h[m]*log(r[m])*sin(theta[m])
            
    
    for j in range(dl):
        
        for k in range(j+dl,N):
            num1=-b[k]*l[k]*(y[j] - y[k])
            num2=b[k]*l[k]*(x[j] - x[k])
            denom=((y[j] - y[k])**2 + (x[j]-x[k])**2 + delta)*n
            dx[j]+=num1/denom
            dy[j]+=num2/denom
             
        for k in range(j+dl):
            xboundary1=np.array([xdown[0],x[0],xup[0]])
            xboundary2=np.append(xboundary1,xup[1:j+dl])
            xboundary3=np.append(xboundary2,np.array([xside6[j+dl],xside2[j+dl],xside4[j+dl],x[j+dl],xside3[j+dl],xside1[j+dl],xside5[j+dl]]))   

            yboundary1=np.array([ydown[0],y[0],yup[0]])            
            yboundary2=np.append(yboundary1,yup[1:j+dl])
            yboundary3=np.append(yboundary2,np.array([yside6[j+dl],yside2[j+dl],yside4[j+dl],y[j+dl],yside3[j+dl],yside1[j+dl],yside5[j+dl]]))           
                        
            xboundary=np.append(xboundary3,np.flipud(xdown[1:j+dl]))
            yboundary=np.append(yboundary3,np.flipud(ydown[1:j+dl]))
            
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
            #print r
            
            for m in range(M):
                if r[m]!=0:       #fix this end points
                   dx[j]+=h[m]*log(r[m])*cos(theta[m])
                   dy[j]+=h[m]*log(r[m])*sin(theta[m])    
    
    for j in range(N-dl,N):
        for k in range(0,j-dl):
            num1=-b[k]*l[k]*(y[j] - y[k])
            num2=b[k]*l[k]*(x[j] - x[k])
            denom=((y[j] - y[k])**2 + (x[j]-x[k])**2)*n
            dx[j]+=num1/denom
            dy[j]+=num2/denom

        for k in range(j-dl,N):
            xboundary1=np.array([xside5[j-dl],xside1[j-dl],xside3[j-dl],x[j-dl],xside4[j-dl],xside2[j-dl],xside6[j-dl]])
            xboundary2=np.append(xboundary1,xup[j-dl:N-1])            
            xboundary3=np.append(xboundary2,np.array([xup[N-1],x[N-1],xdown[N-1]])) 
            
            yboundary1=np.array([yside5[j-dl],yside1[j-dl],yside3[j-dl],y[j-dl],yside4[j-dl],yside2[j-dl],yside6[j-dl]])
            yboundary2=np.append(yboundary1,yup[j-dl:N-1])
            yboundary3=np.append(yboundary2,np.array([yup[N-1],y[N-1],ydown[N-1]]))            
                        
            xboundary=np.append(xboundary3,np.flipud(xdown[j-dl:N-1]))
            yboundary=np.append(yboundary3,np.flipud(ydown[j-dl:N-1]))
            
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
                if r[m]!=0:                    #fix this , end points 
                    dx[j]+=h[m]*log(r[m])*cos(theta[m])
                    dy[j]+=h[m]*log(r[m])*sin(theta[m])
                    
    return np.vstack((0.5*dx/(pi),0.5*dy/(pi)))

@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def curve(x,y,z,b):
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
    
    xmid,ymid=x,y
    x1=np.zeros(N)
    y1=np.zeros(N)
    x2=np.zeros(N)
    y2=np.zeros(N)
    
    for i in range(N):                        
        if z[i]<0:
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
    

def pointremoval(x,y,z,b):
    etol=0.0024
    N=len(x)
    for i in range(2,len(x)-2):
        #if min(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2))<etol:      
        if (np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2))<etol:      
    
            z1=z[:i-1]
            z2=z[i:]
            z=np.append(z1,z2)
        
            x1=x[:i-1]
            x2=x[i:]
            x=np.append(x1,x2)

            y1=y[:i-1]
            y2=y[i:]
            y=np.append(y1,y2)
            
            b1=b[:i-1]
            b2=b[i:]
            b=np.append(b1,b2)
            
    print N-len(x),'removed'
    return x,y,z,b
    
def pointinsertion1(x,y,z,b):    
    N=len(x)-2
    distance=np.ones(N)    
    distance[0:N-1]=((x[1:N] - x[0:N-1])**2 + (y[1:N] - y[0:N-1])**2)**0.5
    distance[N-1]=sqrt((x[N-1] - x[N-2])**2 + (y[N-1] - y[N-2])**2)
    
    maxd=np.max(distance)
    
    etol=0.04
    t=0

    while maxd>etol and t<len(x)*0.5:
        maxarg=np.argmax(distance)
        i=maxarg        
        
        if i==0 and i==1 and i==2:
            i=3
        
        M=np.ones((4,4))
        for j in range(4):
            for k in range(4):
                M[j,k]=z[i+1-j]**k
        
        if np.linalg.det(M)==0:
            print maxd,t            
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
        
        N=len(x)-2
        distance=np.ones(N)    
        distance[0:N-1]=((x[1:N] - x[0:N-1])**2 + (y[1:N] - y[0:N-1])**2)**(0.5)
        distance[N-1]=sqrt((x[N-1] - x[N-2])**2 + (y[N-1] - y[N-2])**2)
    
        maxd=np.max(distance)
        maxarg=np.argmax(distance)
        t=t+1
 
        
    print len(x),'total points'
    return x,y,z,b
    
def pointinsertion(x,y,z,b):
    err=0.04

    diffpx=x[0:len(x)-1]-x[1:len(x)]
    diffpy=y[0:len(y)-1]-y[1:len(y)]
    diffp=np.sqrt(diffpx**2+diffpy**2)

    pmax=np.amax(diffp)
    parg=np.argmax(diffp)

    while pmax>err:
        Mat=np.array(([z[parg-1]**3,z[parg-1]**2,z[parg-1],1.],[z[parg]**3,z[parg]**2,z[parg],1.],[z[parg+1]**3,z[parg+1]**2,z[parg+1],1.],[z[parg+2]**2,z[parg+2]**2,z[parg+2],1.]))

        u1=np.array([x[parg-1],x[parg],x[parg+1],x[parg+2]])
        u2=np.array([y[parg-1],y[parg],y[parg+1],y[parg+2]])
        u3=np.array([b[parg+1],b[parg],b[parg-1],b[parg-2]])

        d1=np.linalg.solve(Mat,u1)
        d2=np.linalg.solve(Mat,u2)
        d3=np.linalg.solve(Mat,u3)

        znew=0.5*(z[parg]+z[parg+1])
        
        xnew=d1[0]*znew**3 + d1[1]*znew**2 + d1[2]*znew + d1[3]
        ynew=d2[0]*znew**3 + d2[1]*znew**2 + d2[2]*znew + d2[3]
        bnew=d3[0]*znew**3 + d3[1]*znew**2 + d3[2]*znew + d3[3]

        x=np.insert(x,parg+1,xnew)
        y=np.insert(y,parg+1,ynew)
        b=np.insert(b,parg+1,bnew)
        
        z=np.insert(z,parg+1,znew)
        diffpx=x[0:len(x)-1]-x[1:len(x)]
        diffpy=y[0:len(y)-1]-y[1:len(y)]
        diffp=np.sqrt(diffpx**2+diffpy**2)
        pmax=np.amax(diffp)
        parg=np.argmax(diffp)
            
                
    return x,y,z,b