# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:06:32 2015

@author: raghavsinghal
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:17:54 2015

@author: raghavsinghal
"""

#from numba import jit,float64
from scipy import interpolate
import numpy as np
from math import cos,pi,sin,sqrt,log
import matplotlib.pyplot as plt

def circle(N,alpha,T):
    N=int(N)
    z=np.arange(N)/(float(N-1.))    
    m=0
    x=(0.5 + 0.05*np.sin(2*m*pi*z))*np.cos(2*pi*z)
    y=(0.5 + 0.05*np.cos(2*m*pi*z))*np.sin(2*pi*z)
    b=0.05*np.ones(N)
    
    delta=0
    etol=0.2*min(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2))
    etolmax=2.4*max(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2))
    
    x,y,z,b=RK4(x,y,z,b,T,delta,etol,alpha,etolmax)        
      
    #xtck=interpolate.splrep(z,x,k=3)
    #ytck=interpolate.splrep(z,y,k=3)

    #x=interpolate.splev(z,xtck)
    #y=interpolate.splev(z,ytck)
    x1,y1,x2,y2=curve(x,y,z,b,0)
    
    
    #plt.figure(2)
    #plt.axis([0,1,-0.25,0.25])
    #plt.title("N=400, b=0.1, |A|=0.2, t=1.2, interval for line integral=5*b")
    plt.plot(x,y,'ro')   
    plt.plot(x2,y2,'ko')    
    plt.plot(x1,y1,'k')
    
    #plt.figure(3)
    #plt.plot(x[0:N/2],y[0:N/2],'bo')
    #plt.plot(x[N/2:],y[N/2:],'r')
    
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
    
#@jit(float64(float64[:],float64[:],float64[:],float64[:],float64,float64,float64,float64))
def RK4(x,y,z,b,T,delta,etol,alpha,etolmax):
    dt=0.2
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
        
        x[N-1],y[N-1]=x[0],y[0]
        

        b=thickness(l,x,y,b)        
        #x,y,z,b=pointinsertion1(x,y,z,b,etolmax)
        #x,y,z,b=pointremoval(x,y,z,b,etol)

        
        #xtck=interpolate.splrep(z,x,np.ones(len(x)),k=5)
        #ytck=interpolate.splrep(z,y,np.ones(len(x)),k=5)

        #xtck=interpolate.splrep(z,x,k=3)  # no smoothing 
        #ytck=interpolate.splrep(z,y,k=3)
    
        #x=interpolate.splev(z,xtck)
        #y=interpolate.splev(z,ytck)
        
        x1,y1,x2,y2=curve(x,y,z,b,1)
       
        x1data=np.append(x1data,x1)   
        y1data=np.append(y1data,y1)

        x2data=np.append(x2data,x2)
        y2data=np.append(y2data,y2)

        xdata=np.append(xdata,x)        
        ydata=np.append(ydata,y)    
        
        Ndata=np.append(Ndata,len(x))
                   
      
        if (Finaltime-t)%1==0:
           print (Finaltime-t),max(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2)),min(np.sqrt((x[0:-2] - x[1:-1])**2 + (y[0:-2] - y[1:-1])**2)),len(x)

    np.savetxt("circlexdata.txt",xdata)
    np.savetxt("circleydata.txt",ydata)
    np.savetxt("circlex1data.txt",x1data)
    np.savetxt("circley1data.txt",y1data)
    np.savetxt("circlex2data.txt",x2data)
    np.savetxt("circley2data.txt",y2data)
    np.savetxt("circleNdata.txt",Ndata)

                

    return x,y,z,b
    
#@jit(float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64,float64))
def F(x,y,w,z,b,delta,alpha):
    N=len(x)
    n=float(N)
    #z=np.linspace(-1,1,N)
    
#    l=np.ones(N)
#    for i in range(1,N-1):
#        l[i]=sqrt(((x[i-1] - x[i+1])*0.5)**2 + ((y[i-1] - y[i+1])*0.5)**2)
#    l[0]=sqrt(((x[0] + x[1])*0.5 - x[0])**2 + ((y[0] + y[1])*0.5 - y[0])**2)
#    l[N-1]=sqrt(((x[N-1] + x[N-2])*0.5 - x[N-1])**2 + ((y[N-1] + y[N-2])*0.5 - y[N-1])**2)
    
#    xtck=interpolate.splrep(z,x,np.ones(len(x)),k=5)
#    ytck=interpolate.splrep(z,y,np.ones(len(x)),k=5)
#    btck=interpolate.splrep(z,b,np.ones(len(x)),k=5)

    xtck=interpolate.splrep(z,x,k=3)
    ytck=interpolate.splrep(z,y,k=3)
    btck=interpolate.splrep(z,b,k=3)
    
    #x=interpolate.splev(z,xtck)
    #y=interpolate.splev(z,ytck)
    #b=interpolate.splev(z,btck)
    
    xdown,ydown,xup,yup=curve(x,y,z,b,1)  
    xside1,yside1,xside2,yside2=curve(x,y,z,b*0.5,0)
    xside3,yside3,xside4,yside4=curve(x,y,z,b*0.25,0)    
    xside5,yside5,xside6,yside6=curve(x,y,z,b*0.75,0)    
    
    xdown[N-1],ydown[N-1]=xdown[0],ydown[0]
    xup[N-1],yup[N-1]=xdown[0],ydown[0]
    xside1[N-1],yside1[N-1]=xside1[0],yside1[0]
    xside2[N-1],yside2[N-1]=xside2[0],yside2[0]
    xside3[N-1],yside3[N-1]=xside3[0],yside3[0]
    xside4[N-1],yside4[N-1]=xside4[0],yside4[0]
    xside5[N-1],yside5[N-1]=xside5[0],yside5[0]
    xside6[N-1],yside6[N-1]=xside6[0],yside6[0]
    
    
    xdowntck=interpolate.splrep(z,xdown,np.ones(len(x)),k=5)
    ydowntck=interpolate.splrep(z,ydown,np.ones(len(x)),k=5)

    xuptck=interpolate.splrep(z,xdown,np.ones(len(x)),k=5)
    yuptck=interpolate.splrep(z,ydown,np.ones(len(x)),k=5)

    
    x=np.append(x,np.append(x,x))
    y=np.append(y,np.append(y,y))
    
    xdown=np.append(xdown,np.append(xdown,xdown))
    ydown=np.append(ydown,np.append(ydown,ydown))
    
    xup=np.append(xup,np.append(xup,xup))
    yup=np.append(yup,np.append(yup,yup))
        
    xside1=np.append(xside1,np.append(xside1,xside1))
    yside1=np.append(yside1,np.append(yside1,yside1))    

    xside2=np.append(xside2,np.append(xside2,xside2))
    yside2=np.append(yside2,np.append(yside2,yside2))    
    
    xside3=np.append(xside3,np.append(xside3,xside3))
    yside3=np.append(yside3,np.append(yside3,yside3))    

    xside4=np.append(xside4,np.append(xside4,xside4))
    yside4=np.append(yside4,np.append(yside4,yside4))    

    xside5=np.append(xside5,np.append(xside5,xside5))
    yside5=np.append(yside5,np.append(yside5,yside5))        
    
    xside6=np.append(xside6,np.append(xside6,xside6))
    yside6=np.append(yside6,np.append(yside6,yside6))    
    
    dx=np.zeros(N)
    dy=np.zeros(N)       
    

    #xmid=np.append(x[0],np.append((x[0:N-1] + x[1:N])*0.5,x[N-1]))
    #ymid=np.append(y[0],np.append((y[0:N-1] + y[1:N])*0.5,y[N-1]))   
            
    for j in range(N):
        t=0        
        length=0
        
        while length<=b[j]*alpha :
              for a in range(t):
                  length+=sqrt((x[j+N]-x[j+N+a])**2 + (y[j+N] - y[j+N+a])**2)
        
              t=t+1
        
        s=0
        length=0
        while length<=b[j-N]*alpha:
             for a in range(s):
                 length+=sqrt((x[j+N]-x[j+N-a])**2 + (y[j+N] - y[j+N-a])**2)        
             s=s+1
        

        if j-s>0 and j+t<N:
           M=3*len(z[:j-s])
           znew=np.linspace(z[0],z[j-s],M) 
           xnew=interpolate.splev(znew,xtck)
           ynew=interpolate.splev(znew,ytck)
           bnew=interpolate.splev(znew,btck)
           
           l=np.ones(M)
           for i in range(1,M-1):
               l[i]=sqrt(((xnew[i-1] - xnew[i+1])*0.5)**2 + ((ynew[i-1] - ynew[i+1])*0.5)**2)
           l[0]=sqrt(((xnew[0] + xnew[1])*0.5 - xnew[0])**2 + ((ynew[0] + ynew[1])*0.5 - ynew[0])**2)
           l[M-1]=sqrt(((xnew[M-1-1] + xnew[M-1-2])*0.5 - xnew[M-1-1])**2 + ((ynew[M-1-1] + ynew[M-1-2])*0.5 - ynew[M-1-1])**2)

           for k in range(2*len(z[:j-s])):
               num1=-((y[j] - ynew[k]))
               num2=((x[j] - xnew[k]))
               denom=(((y[j] - ynew[k])**2) + ((x[j]-xnew[k])**2) )*n
               dx[j]+=num1/denom
               dy[j]+=num2/denom

           znew=np.linspace(z[j+t],z[N-1],2*len(z[j+t:])) 
           xnew=interpolate.splev(znew,xtck)
           ynew=interpolate.splev(znew,ytck)
           bnew=interpolate.splev(znew,btck)

           M=3*len(z[j+t:])
           znew=np.linspace(z[j+t],z[-1],M) 
           xnew=interpolate.splev(znew,xtck)
           ynew=interpolate.splev(znew,ytck)
           bnew=interpolate.splev(znew,btck)
           
           l=np.ones(M)
           for i in range(1,M-1):
               l[i]=sqrt(((xnew[i-1] - xnew[i+1])*0.5)**2 + ((ynew[i-1] - ynew[i+1])*0.5)**2)
           l[0]=sqrt(((xnew[0] + xnew[1])*0.5 - xnew[0])**2 + ((ynew[0] + ynew[1])*0.5 - ynew[0])**2)
           l[M-1]=sqrt(((xnew[M-1-1] + xnew[M-1-2])*0.5 - xnew[M-1-1])**2 + ((ynew[M-1-1] + ynew[M-1-2])*0.5 - ynew[M-1-1])**2)
 
           for k in range(2*len(z[j+t:])):
               num1=-((y[j] - ynew[k]))
               num2=((x[j] - xnew[k]))
               denom=(((y[j] - ynew[k])**2) + ((x[j]-xnew[k])**2) )*n
               dx[j]+=num1/denom
               dy[j]+=num2/denom
               
        elif j+t>N and j-s>0:
           M=3*len(z[j+t-N:j-s])
           znew=np.linspace(z[j+t-N],z[j-s],M) 
           xnew=interpolate.splev(znew,xtck)
           ynew=interpolate.splev(znew,ytck)
           bnew=interpolate.splev(znew,btck)

           l=np.ones(M)
           for i in range(1,M-1):
               l[i]=sqrt(((xnew[i-1] - xnew[i+1])*0.5)**2 + ((ynew[i-1] - ynew[i+1])*0.5)**2)
           l[0]=sqrt(((xnew[0] + xnew[1])*0.5 - xnew[0])**2 + ((ynew[0] + ynew[1])*0.5 - ynew[0])**2)
           l[M-1]=sqrt(((xnew[M-1-1] + xnew[M-1-2])*0.5 - xnew[M-1-1])**2 + ((ynew[M-1-1] + ynew[M-1-2])*0.5 - ynew[M-1-1])**2)

           for k in range(2*len(z[j+t-N:j-s])):
               num1=-((y[j] - ynew[k]))
               num2=((x[j] - xnew[k]))
               denom=(((y[j] - ynew[k])**2) + ((x[j]-xnew[k])**2) )*n
               dx[j]+=num1/denom
               dy[j]+=num2/denom

        elif j+t<N and j-s<0:
           M=3*len(z[j+t:j-s+N])
           znew=np.linspace(z[j+t],z[j-s+N],M) 
           xnew=interpolate.splev(znew,xtck)
           ynew=interpolate.splev(znew,ytck)
           bnew=interpolate.splev(znew,btck)

           l=np.ones(M)
           for i in range(1,M-1):
               l[i]=sqrt(((xnew[i-1] - xnew[i+1])*0.5)**2 + ((ynew[i-1] - ynew[i+1])*0.5)**2)
           l[0]=sqrt(((xnew[0] + xnew[1])*0.5 - xnew[0])**2 + ((ynew[0] + ynew[1])*0.5 - ynew[0])**2)
           l[M-1]=sqrt(((xnew[M-1-1] + xnew[M-1-2])*0.5 - xnew[M-1-1])**2 + ((ynew[M-1-1] + ynew[M-1-2])*0.5 - ynew[M-1-1])**2)

           for k in range(2*len(z[j+t:j-s+N])):
               num1=-((y[j] - ynew[k]))
               num2=((x[j] - xnew[k]))
               denom=(((y[j] - ynew[k])**2) + ((x[j]-xnew[k])**2) )*n
               dx[j]+=num1/denom
               dy[j]+=num2/denom

        
        
        xboundary1=np.array([xside5[j-s+N],xside1[j-s+N],xside3[j-s+N],x[j-s+N],xside4[j-s+N],xside2[j-s+N],xside6[j-s+N]])
        if j+t<N and j-s>0:
            
           zup=np.linspace(z[j-s],z[j+t],200)
           xupnew=interpolate.splev(zup,xuptck)
           yupnew=interpolate.splev(zup,yuptck)
        
        elif j+t>N and j-s>0:
            zup1=np.linspace(z[j-s],z[-1],80)
            zup2=np.linspace(z[0],z[j+t-N],80)
            
            xupnew1=interpolate.splev(zup1,xuptck)
            yupnew1=interpolate.splev(zup1,yuptck)
            xupnew2=interpolate.splev(zup2,xuptck)
            yupnew2=interpolate.splev(zup2,yuptck)
            
            xupnew=np.append(xupnew1,xupnew2)
            yupnew=np.append(yupnew1,yupnew2)
        
        elif j+t<N and j-s<0:
            zup1=np.linspace(z[j-s+N],z[-1],80)
            zup2=np.linspace(z[0],z[j+t],80)
            
            xupnew1=interpolate.splev(zup1,xuptck)
            yupnew1=interpolate.splev(zup1,yuptck)
            xupnew2=interpolate.splev(zup2,xuptck)
            yupnew2=interpolate.splev(zup2,yuptck)
            
            xupnew=np.append(xupnew1,xupnew2)
            yupnew=np.append(yupnew1,yupnew2)
            
        xboundary2=np.append(xboundary1,xupnew)        
        xboundary3=np.append(xboundary2,np.array([xside6[j+t+N],xside2[j+t+N],xside4[j+t+N],x[j+t+N],xside3[j+t+N],xside1[j+t+N],xside5[j+t+N]]))   
            
            
        yboundary1=np.array([yside5[j-s+N],yside1[j-s+N],yside3[j-s+N],y[j-s+N],yside4[j-s+N],yside2[j-s+N],yside6[j-s+N]])
        yboundary2=np.append(yboundary1,yupnew)
        yboundary3=np.append(yboundary2,np.array([yside6[j+t+N],yside2[j+t+N],yside4[j+t+N],y[j+t+N],yside3[j+t+N],yside1[j+t+N],yside5[j+t+N]]))           

        if j+t<N and j-s>0:
            
           zdown=np.linspace(z[j-s],z[j+t],200)
           xdownnew=interpolate.splev(zdown,xdowntck)
           ydownnew=interpolate.splev(zdown,ydowntck)
        
        elif j+t>N and j-s>0:
            zdown1=np.linspace(z[j-s],z[-1],80)
            zdown2=np.linspace(z[0],z[j+t-N],80)
            
            xdownnew1=interpolate.splev(zdown1,xdowntck)
            ydownnew1=interpolate.splev(zdown1,ydowntck)
            xdownnew2=interpolate.splev(zdown2,xdowntck)
            ydownnew2=interpolate.splev(zdown2,ydowntck)
            
            xdownnew=np.append(xdownnew1,xdownnew2)
            ydownnew=np.append(ydownnew1,ydownnew2)
        
        elif j+t<N and j-s<0:
            zdown1=np.linspace(z[j-s+N],z[-1],80)
            zdown2=np.linspace(z[0],z[j+t],80)
            
            xdownnew1=interpolate.splev(zdown1,xdowntck)
            ydownnew1=interpolate.splev(zdown1,ydowntck)
            xdownnew2=interpolate.splev(zdown2,xdowntck)
            ydownnew2=interpolate.splev(zdown2,ydowntck)
            
            xdownnew=np.append(xdownnew1,xdownnew2)
            ydownnew=np.append(ydownnew1,ydownnew2)
                        
        xboundary=np.append(xboundary3,np.flipud(xdownnew))
        yboundary=np.append(yboundary3,np.flipud(ydownnew))
        
    

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
            dx[j]+=h[m]*log(r[m])*cos(theta[m])/pi            
            dy[j]+=h[m]*log(r[m])*sin(theta[m])/pi
    
    #plt.figure(2)
    #plt.plot(xboundary,yboundary,'bo')
            
    #xnew1=interpolate.splev(z[:N-1-s],xtck)
    #ynew1=interpolate.splev(z[:N-1-s],ytck)

        #plt.figure(2)
    #plt.plot(xnew1,ynew1,'r')            
    #plt.plot(xboundary,yboundary,'k')


    return np.vstack((0.5*dx,0.5*dy))

    

#@jit(float64(float64[:],float64[:],float64[:],float64[:],float64))
def curve(x,y,z,b,integral):
    N=len(x)
    
    if integral==1:
        xmid=np.append(x[0],np.append((x[0:N-1] + x[1:N])*0.5,x[N-1]))
        ymid=np.append(y[0],np.append((y[0:N-1] + y[1:N])*0.5,y[N-1]))   
    else:
        xmid,ymid,z=x,y,z
        
    slope=np.zeros(N)
    for i in range(3,N-1):
        A=np.array([[z[i-3]**4,z[i-3]**3,z[i-3]**2,z[i-3],1.],[z[i-2]**4,z[i-2]**3,z[i-2]**2,z[i-2],1.],[z[i-1]**4,z[i-1]**3,z[i-1]**2,z[i-1],1.],[z[i]**4,z[i]**3,z[i]**2,z[i],1.],[z[i+1]**4,z[i+1]**3,z[i+1]**2,z[i+1],1.]])
        u=np.array([xmid[i-3],xmid[i-2],xmid[i-1],xmid[i],xmid[i+1]])
        v=np.array([ymid[i-3],ymid[i-2],ymid[i-1],ymid[i],ymid[i+1]])
        
        a=np.linalg.solve(A,u)
        c=np.linalg.solve(A,v)
        
        slope[i-3]=(4*c[0]*z[i-3]**3 + 3*c[1]*z[i-3]**2 +2*c[2]*z[i-3] + c[3])/(4*a[0]*z[i-3]**3 + 3*a[1]*z[i-3]**2 +2*a[2]*z[i-3] + a[3])
        slope[i-2]=(4*c[0]*z[i-2]**3 + 3*c[1]*z[i-2]**2 +2*c[2]*z[i-2] + c[3])/(4*a[0]*z[i-2]**3 + 3*a[1]*z[i-2]**2 +2*a[2]*z[i-2] + a[3])
        slope[i-1]=(4*c[0]*z[i-1]**3 + 3*c[1]*z[i-1]**2 +2*c[2]*z[i-1] + c[3])/(4*a[0]*z[i-1]**3 + 3*a[1]*z[i-1]**2 +2*a[2]*z[i-1] + a[3])
        slope[i]=(4*c[0]*z[i]**3 + 3*c[1]*z[i]**2 +2*c[2]*z[i] + c[3])/(4*a[0]*z[i]**3 + 3*a[1]*z[i]**2 +2*a[2]*z[i] + a[3])
        slope[i+1]=(4*c[0]*z[i+1]**3 + 3*c[1]*z[i+1]**2 +2*c[2]*z[i+1] + c[3])/(4*a[0]*z[i+1]**3 + 3*a[1]*z[i+1]**2 +2*a[2]*z[i+1] + a[3])
         
    
    x1=np.zeros(N)
    y1=np.zeros(N)
    x2=np.zeros(N)
    y2=np.zeros(N)
    
    for i in range(N):                        

        if z[i]>=0:
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
    
    
    x1tck=interpolate.splrep(z,x1,k=3)
    y1tck=interpolate.splrep(z,y1,k=3)
    x2tck=interpolate.splrep(z,x2,k=3)
    y2tck=interpolate.splrep(z,y2,k=3)
    
    x1=interpolate.splev(z,x1tck)
    y1=interpolate.splev(z,y1tck)
    x2=interpolate.splev(z,x2tck)
    y2=interpolate.splev(z,y2tck)
    
    return x1,y1,x2,y2

def pointremoval(x,y,z,b,etol):
    N=len(x)

    diffpx=np.ones(len(x))
    diffpy=np.ones(len(x))
    N=len(x)
    diffpx[0:N-1]=x[0:len(x)-1]-x[1:len(x)]
    diffpy[0:N-1]=y[0:len(y)-1]-y[1:len(y)]
    diffpx[N-1]=x[0]-x[N-1]
    diffpy[N-1]=y[0]-y[N-1]
    diffp=np.sqrt(diffpx**2+diffpy**2)

    pmin=np.amin(diffp)
    parg=np.argmin(diffp)
    
    X,Y,Z,B=x,y,z,b
    t=0
    while pmin<etol and t<10:
    #for i in range(N-1):
            i=parg
        #if sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)<etol:
            
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
            
            diffpx=np.ones(len(X))
            diffpy=np.ones(len(X))
            N=len(X)
            diffpx[0:N-1]=X[0:len(X)-1]-X[1:len(Y)]
            diffpy[0:N-1]=Y[0:len(Y)-1]-Y[1:len(Y)]
            diffpx[N-1]=X[0]-X[N-1]
            diffpy[N-1]=Y[0]-Y[N-1]
            diffp=np.sqrt(diffpx**2+diffpy**2)
            pmin=np.amin(diffp)
            parg=np.argmin(diffp)
            t=t+1
            
    print 'removed point',len(x)-len(X)        
    return X,Y,Z,B

#@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def pointinsertion1(x,y,z,b,err):
    #err=0.02
    M=len(x)
    diffpx=np.ones(len(x))
    diffpy=np.ones(len(x))
    N=len(x)
    diffpx[0:N-1]=x[0:len(x)-1]-x[1:len(x)]
    diffpy[0:N-1]=y[0:len(y)-1]-y[1:len(y)]
    diffpx[N-1]=x[0]-x[N-1]
    diffpy[N-1]=y[0]-y[N-1]
    diffp=np.sqrt(diffpx**2+diffpy**2)

    pmax=np.amax(diffp)
    parg=np.argmax(diffp)
    t=0


    while pmax>err and t<10:
        xtck=interpolate.splrep(z,x,k=3)
        ytck=interpolate.splrep(z,y,k=3)
        btck=interpolate.splrep(z,b,k=3)
        
        if parg==len(x)-1:
            znew=(z[len(x)-1]+1)*0.5
        else:
            znew=0.5*(z[parg]+z[parg+1])
        
        xnew=interpolate.splev(znew,xtck)
        ynew=interpolate.splev(znew,ytck)
        bnew=interpolate.splev(znew,btck)
        #ynew=d2[0]*znew**3 + d2[1]*znew**2 + d2[2]*znew + d2[3]
        #bnew=d3[0]*znew**3 + d3[1]*znew**2 + d3[2]*znew + d3[3]

        x=np.insert(x,parg+1,xnew)
        y=np.insert(y,parg+1,ynew)
        b=np.insert(b,parg+1,bnew)        
        z=np.insert(z,parg+1,znew)
        
        diffpx=np.ones(len(x))
        diffpy=np.ones(len(x))
        N=len(x)
        diffpx[0:N-1]=x[0:len(x)-1]-x[1:len(x)]
        diffpy[0:N-1]=y[0:len(y)-1]-y[1:len(y)]
        diffpx[N-1]=x[0]-x[N-1]
        diffpy[N-1]=y[0]-y[N-1]
        diffp=np.sqrt(diffpx**2+diffpy**2)
        pmax=np.amax(diffp)
        parg=np.argmax(diffp)
        t=t+1
    print 'points added',len(x)-M
        
    return x,y,z,b        
    