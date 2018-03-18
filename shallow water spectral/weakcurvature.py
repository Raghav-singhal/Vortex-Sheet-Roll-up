# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:32:42 2015

@author: raghavsinghal
"""

from numba import jit,float64
import numpy as np
from math import sin,cos,pi,sqrt,cosh,sinh
import matplotlib.pyplot as plt

@jit(float64(float64,float64,float64))
def at3(N,delta,T):
    n=float(N)
    z =np.arange(N)/(n)
    
    y = -0.01*np.sin(2*pi*z)
    x = z + 0.01*np.sin(2*pi*z)
    
    x,y,b = RK4(x,y,z,T,delta) # x and y after integrating, b is the updated thickness
    
    a=plots(x,y,z)
    x1,y1,x2,y2=curve(x,y,z,b)
    
    plt.figure(1)
    plt.title("Evolution of curve",fontsize=20)
    plt.plot(x,y,'bo',lw=2)
    plt.plot(x1,y1,'ro',lw=2)
    plt.plot(x2,y2,'ro',lw=2)
    plt.axis([0,1,-0.15,0.15])
    plt.xlabel('x',fontsize=20)
    plt.ylabel("y", fontsize=20) 
    
    plt.figure(5)    
    plt.title("curvature")
    plt.plot(z,a,'k')
   
    ddx,ddy,tolerance=np.zeros(N),np.zeros(N),np.zeros(N)     
    for i in range(N):
        if z[i]<0.5:
            ddx[i]=((x[i+1]-x[i])/(z[i+1]-z[i]) - (x[i]-x[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
            ddy[i]=((y[i+1]-y[i])/(z[i+1]-z[i]) - (y[i]-y[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
        elif z[i]>0.5:
            ddx[i]=((x[i]-x[i-1])/(z[i]-z[i-1]) - (x[i-1]-x[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
            ddy[i]=((y[i]-y[i-1])/(z[i]-z[i-1]) - (y[i-1]-y[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
    ddx[0]=ddx[N-1]
    ddy[0]=ddy[N-1]
              

    for i in range(N):
        if z[i]<0.5:
            tolerance[i]=(ddx[i+1]-ddx[i])/(z[i+1]-z[i])
        if z[i]>0.5:
            tolerance[i]=(ddx[i]-ddx[i-1])/(z[i]-z[i-1])
    for i in range(N):
        if z[i]==0.5:
            tolerance[i]=(tolerance[i-1] + tolerance[i+1])*0.5
    plt.figure(22)
    plt.title('ypp')
    plt.plot(z,ddy)
#    plt.title('tolerance')
#    plt.plot(z,tolerance)
               

@jit(float64(float64[:],float64[:],float64[:],float64[:],float64,float64,float64[:]))
def rotation(x,y,z,ddx,minp,maxp,ddy):
    N=len(x)
    centre=0
    start=0
    end=0
    if (start+end)%2==0:
        centre=int((start+end)*0.5)
        start=int(minp -2)
        end=int(maxp + 3)
    else:
        centre=int((start+end+1)*0.5)
        start=int(minp-2)
        end=int(maxp + 4)
    
    for i in range(start,centre):
        M=np.matrix([[z[i]**4,z[i]**3,z[i]**2,z[i],1],[z[i+1]**4 , z[i+1]**3, z[i+1]**2, z[i+1] ,1],[12*z[i+2]**2 , 6*z[i+2] , 2., 0. , 0.],[12*z[i+3]**2 , 6*z[i+3] , 2., 0. , 0.],[12*z[i+4]**2 , 6*z[i+4] , 2., 0. , 0.]])
        
        u=np.array([x[i],x[i+1],ddx[i+2],ddx[i+3],ddx[i+4]])
        v=np.array([y[i],y[i+1],ddy[i+2],ddy[i+3],ddy[i+4]])

        a=np.linalg.solve(M,u)
        b=np.linalg.solve(M,v)

        x[i+2]=a[0]*z[i+2]**4 + a[1]*z[i+2]**3 + a[2]*z[i+2]**2 + a[3]*z[i+2] + a[4]
        x[i+3]=a[0]*z[i+3]**4 + a[1]*z[i+3]**3 + a[2]*z[i+3]**2 + a[3]*z[i+3] + a[4]
        x[i+4]=a[0]*z[i+4]**4 + a[1]*z[i+4]**3 + a[2]*z[i+4]**2 + a[3]*z[i+4] + a[4]

        y[i+2]=b[0]*z[i+2]**4 + b[1]*z[i+2]**3 + b[2]*z[i+2]**2 + b[3]*z[i+2] + b[4]
        y[i+3]=b[0]*z[i+3]**4 + b[1]*z[i+3]**3 + b[2]*z[i+3]**2 + b[3]*z[i+3] + b[4]
        y[i+4]=b[0]*z[i+4]**4 + b[1]*z[i+4]**3 + b[2]*z[i+4]**2 + b[3]*z[i+4] + b[4]
    for i in range(centre):
        M=np.matrix([[z[end-i]**4,z[end-i]**3,z[end-i]**2,z[end-i],1.],[z[end-i-1]**4,z[end-i-1]**3,z[end-i-1]**2,z[end-i-1],1.],[12*z[end-i-2]**2,6*z[end-i-2],2.,0.,0.],[12*z[end-i-3]**2,6*z[end-i-3],2.,0.,0.],[12*z[end-i-4]**2,6*z[end-i-4],2.,0.,0.]])
        u=np.array([x[end-i],x[end-i-1],ddx[end-i-2],ddx[end-i-3],ddx[end-i-4]])
        v=np.array([y[end-i],y[end-i-1],ddy[end-i-2],ddy[end-i-3],ddy[end-i-4]])
        
        a=np.linalg.solve(M,u)
        b=np.linalg.solve(M,v)
        
        x[end-i-2]=a[0]*z[end-i-2]**4 + a[1]*z[end-i-2]**3 + a[2]*z[end-i-2]**2 + a[3]*z[end-i-2] + a[4]
        x[end-i-3]=a[0]*z[end-i-3]**4 + a[1]*z[end-i-3]**3 + a[2]*z[end-i-3]**2 + a[3]*z[end-i-3] + a[4]
        x[end-i-4]=a[0]*z[end-i-4]**4 + a[1]*z[end-i-4]**3 + a[2]*z[end-i-4]**2 + a[3]*z[end-i-4] + a[4]
        
        y[end-i-2]=b[0]*z[end-i-2]**4 + b[1]*z[end-i-2]**3 + b[2]*z[end-i-2]**2 + b[3]*z[end-i-2] + b[4]
        y[end-i-3]=b[0]*z[end-i-3]**4 + b[1]*z[end-i-3]**3 + b[2]*z[end-i-3]**2 + b[3]*z[end-i-3] + b[4]
        y[end-i-4]=b[0]*z[end-i-4]**4 + b[1]*z[end-i-4]**3 + b[2]*z[end-i-4]**2 + b[3]*z[end-i-4] + b[4]
    
    xhat=np.fft.rfft(x-z)
    yhat=np.fft.rfft(y)

    for i in range(N/2 + 1):
        if  abs(xhat[i])<1e-6:
            xhat[i]=0.0
        if abs(yhat[i])<1e-6:
            yhat[i]=0.    
    
    x=np.fft.irfft(xhat) + z
    y=np.fft.irfft(yhat)
    
    return x,y

@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def thickness(l,x,y,b):
    N=len(x)
    x2mid=(x[0:N-1] - x[1:N])
    y2mid=(y[0:N-1] - y[1:N])
    lnew=np.sqrt(x2mid**2 + y2mid**2)
    b=b*l/lnew
    return b
    
@jit(float64(float64[:],float64[:],float64,))
def f(x,y,delta):
    N=len(x)
    n=float(N)
    dx=np.zeros(N)
    dy=np.zeros(N)

    for j in range(N):
        for k in range(N):
            if k!=j:
               xnum = sinh(2*pi*(y[j]-y[k]))
               ynum = sin(2*pi*(x[j]-x[k]))
               denom = cosh(2*pi*(y[j]-y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2
               dx[j] += xnum/denom
               dy[j] += ynum/denom
    return np.vstack((-0.5*dx/n,0.5*dy/n))

@jit(float64(float64[:],float64[:],float64,float64,float64))
def shock(x,y,delta,maxp,minp):
    N=len(x)
    n=float(N)
    dx=np.zeros(N)
    dy=np.zeros(N)
    
    maxp=int(maxp)
    minp=int(minp)
    
    for j in range(N):     
        for k in range(N):
            if k!=j:
               xnum = sinh(2*pi*(y[j]-y[k]))
               ynum = sin(2*pi*(x[j]-x[k]))
               denom = cosh(2*pi*(y[j]-y[k])) - cos(2*pi*(x[j]-x[k])) + delta**2
               dx[j] += xnum/denom
               dy[j] += ynum/denom
    dx[minp:maxp]=np.zeros(maxp-minp)
    dy[minp:maxp]=np.zeros(maxp-minp)
    return np.vstack((-0.5*dx/n,0.5*dy/n))
    

@jit(float64(float64[:],float64[:],float64[:],float64,float64))
def RK4(x,y,z,T,delta):
    N=len(x)
    dt=0.01
    Finaltime=int(T/dt)
    b=0.1*np.ones(N-1) 

    ddx=np.zeros(N)
    ddy=np.zeros(N)
    maxp=0
    minp=0
    tolerance=np.zeros(N)
    t=0
    etol=100
    
    for i in range(N):
        if z[i]<0.5:
            ddx[i]=((x[i+1]-x[i])/(z[i+1]-z[i]) - (x[i]-x[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
            ddy[i]=((y[i+1]-y[i])/(z[i+1]-z[i]) - (y[i]-y[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
        elif z[i]>0.5:
            ddx[i]=((x[i]-x[i-1])/(z[i]-z[i-1]) - (x[i-1]-x[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
            ddy[i]=((y[i]-y[i-1])/(z[i]-z[i-1]) - (y[i-1]-y[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
    for i in range(N):
        if z[i]==0.5:
            ddx[i]=(ddx[i+1]+ddx[i-1])*0.5
            ddy[i]=(ddy[i+1]+ddy[i-1])*0.5

           
    ddx[0]=ddx[N-1]
    ddy[0]=ddy[N-1]
           
    
    xdata,ydata=x,y 
    curvaturedata=plots(x,y,z)
    x1data,y1data,x2data,y2data=curve(x,y,z,b)
    zdata=z
    ddxdata,ddydata=ddx,ddy
    
    while t<=Finaltime:
        if np.max(np.abs(tolerance))<etol:  
           if (Finaltime-t)%10==0:
              print (Finaltime-t)
              
           xmid = (x[0:N-1] - x[1:N])
           ymid = (y[0:N-1] - y[1:N])
           l = np.sqrt(xmid*xmid + ymid*ymid)
        
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
    
#           xhat=np.fft.rfft(x-z)
#           yhat=np.fft.rfft(y)
#        
#
#           for i in range(N/2 + 1):
#               if  abs(xhat[i])<1e-13:
#                   xhat[i]=0.0
#               if abs(yhat[i])<1e-13:
#                   yhat[i]=0.
#                
#           x=np.fft.irfft(xhat) + z
#           y=np.fft.irfft(yhat)

           for i in range(N):
               if z[i]<0.5:
                  ddx[i]=((x[i+1]-x[i])/(z[i+1]-z[i]) - (x[i]-x[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
                  ddy[i]=((y[i+1]-y[i])/(z[i+1]-z[i]) - (y[i]-y[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
               elif z[i]>0.5:
                  ddx[i]=((x[i]-x[i-1])/(z[i]-z[i-1]) - (x[i-1]-x[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
                  ddy[i]=((y[i]-y[i-1])/(z[i]-z[i-1]) - (y[i-1]-y[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
           for i in range(N):
               if z[i]==0.5:
                   ddx[i]=(ddx[i+1]+ddx[i-1])*0.5
                   ddy[i]=(ddy[i+1]+ddy[i-1])*0.5

           ddx[0]=ddx[N-1]
           ddy[0]=ddy[N-1]
           
           xppmax=np.max(ddx)
           xppmin=np.min(ddx)
           
           yppmax=np.max(ddy)
           yppmin=np.min(ddy)



           for i in range(N):
               if z[i]<0.5:
                   tolerance[i]=(ddx[i+1]-ddx[i])/(z[i+1]-z[i])
               if z[i]>0.5:
                   tolerance[i]=(ddx[i]-ddx[i-1])/(z[i]-z[i-1])
           for i in range(N):
               if z[i]==0.5:
                   tolerance[i]=(tolerance[i-1] + tolerance[i+1])*0.5
           
           #a=np.array([])
           #for i in range(N-1):
           #    if abs(tolerance[i])>=40:
           #       a=np.append(a,i)
#           curvature1=plots(x,y,z)
#           x1,y1,x2,y2=curve(x,y,z,b)
#           curvaturedata=np.append(curvaturedata,curvature1)
#     
#           zdata=np.append(zdata,z)
#           x1data=np.append(x1data,x1)   
#           y1data=np.append(y1data,y1)
#           x2data=np.append(x2data,x2)
#           y2data=np.append(y2data,y2)
#           xdata=np.append(xdata,x)        
#           ydata=np.append(ydata,y)
#           ddxdata=np.append(ddxdata,ddx)
#           ddydata=np.append(ddydata,ddy)
           #maxp=max(a)
           #minp=min(a)

           b=thickness(l,x,y,b)  
           t=t+1
        else:
            break
        
#loop changes here

    maxp=np.argmax(ddx)
    minp=np.argmin(ddx)

    #maxp=int(max(a))
    #minp=int(min(a))
    print maxp,minp

    for t in range(t,Finaltime):
        
        if (Finaltime-t)%10==0:
            print Finaltime-t,minp

        xmid = (x[0:N-1] - x[1:N])
        ymid = (y[0:N-1] - y[1:N])
        l = np.sqrt(xmid*xmid + ymid*ymid)
      
        F1= shock(x,y,delta,maxp,minp)
        xk1,yk1 = dt*F1[0],dt*F1[1]
    
        F2= shock(x + 0.5*xk1,y + 0.5*yk1, delta,maxp,minp)
        xk2 = dt*F2[0]
        yk2 = dt*F2[1]
    
        F3=shock(x + 0.5*xk2,y + 0.5*yk2,delta,maxp,minp)
        xk3 = dt*F3[0]
        yk3 = dt*F3[1]
    
        F4=shock(x + xk3,y+ yk3,delta,maxp,minp)
        xk4 = dt*F4[0]
        yk4 = dt*F4[1]
    
        x = x + (1./6.)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        y = y + (1./6.)*(yk1 + 2*yk2 + 2*yk3 + yk4)
          
        ddxnew=np.zeros(N)
        ddynew=np.zeros(N)
        #ddxnew[minp:maxp]=ddx[minp:maxp]
        #ddynew[minp:maxp]=ddy[minp:maxp]

        
        for i in range(N):
            if i<minp:
                ddxnew[i]=((x[i+1]-x[i])/(z[i+1]-z[i]) - (x[i]-x[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
                ddynew[i]=((y[i+1]-y[i])/(z[i+1]-z[i]) - (y[i]-y[i-1])/(z[i]-z[i-1]))/(z[i+1]-z[i])
            elif i>maxp:
                ddxnew[i]=((x[i]-x[i-1])/(z[i]-z[i-1]) - (x[i-1]-x[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
                ddynew[i]=((y[i]-y[i-1])/(z[i]-z[i-1]) - (y[i-1]-y[i-2])/(z[i-1]-z[i-2]))/(z[i]-z[i-1])
        
        
        #for j in range(N):
        #    if z[j]==0.5:
        #        ddxnew[j]=(ddxnew[j+1]+ddxnew[j-1])*0.5
        #        ddynew[j]=(ddynew[j+1]+ddynew[j-1])*0.5                
        ddxnew[0]=ddxnew[N-1]
        ddynew[0]=ddynew[N-1]
        

        #xhat=np.fft.rfft(x-z)
        #yhat=np.fft.rfft(y)
                    
#        x=np.fft.irfft(xhat) + z
#        y=np.fft.irfft(yhat)          
#        
#        xhat=np.fft.rfft(x-z)
#        yhat=np.fft.rfft(y)
#        ik=1j*np.hstack((np.arange(0,N/2 +1)))
#
#        ddxnew=ik*ik*xhat
#        ddynew=ik*ik*yhat
#        
#        ddxnew=np.fft.irfft(ddxnew)
#        ddynew=np.fft.irfft(ddynew)
        
        c=np.array([])
        d=np.array([])
        e=np.array([])
        for i in range(N):
            if ddxnew[i]>=xppmax:
                ddxnew[i]=xppmax
                c=np.append(c,i)
                d=np.append(d,i)
        
        for i in range(N):        
            if ddxnew[i]<=xppmin:
                ddxnew[i]=xppmin
                c=np.append(c,i)                
                e=np.append(e,i)
        for i in range(N):
            if ddynew[i]>=yppmax:
                ddynew[i]=yppmax
         #       d=np.append(d,i)
        
        for i in range(N):        
            if ddynew[i]<=yppmin:
                ddynew[i]=yppmin            
         #       d=np.append(d,i)
        
        minp=int(min(c))
        maxp=int(max(c))

        ddxnew[minp:maxp]=ddx[minp:maxp]
        ddynew[minp:maxp]=ddy[minp:maxp]


        #ddxnew[int(min(e)):int(max(e))]=np.ones(int(max(e))-int(min(e)))*xppmin
        #ddxnew[int(min(d)):int(max(d))]=np.ones(int(max(d))-int(min(d)))*xppmax
        

        ddx,ddy=ddxnew,ddynew
        
        #print minp,maxp
        if Finaltime-t==1:
           plt.figure(17)
           plt.title('modified xpp')
           #plt.plot(z[minp:maxp],ddxnew[minp:maxp])
           plt.scatter(z,ddx)           
           
        x,y=rotation(x,y,z,ddxnew,minp,maxp,ddynew)
        curvature1=plots(x,y,z)
        x1,y1,x2,y2=curve(x,y,z,b)
        curvaturedata=np.append(curvaturedata,curvature1)
        ddxdata=np.append(ddxdata,ddx)
        ddydata=np.append(ddydata,ddy)

        zdata=np.append(zdata,z)
        x1data=np.append(x1data,x1)   
        y1data=np.append(y1data,y1)
        x2data=np.append(x2data,x2)
        y2data=np.append(y2data,y2)
        xdata=np.append(xdata,x)        
        ydata=np.append(ydata,y)  

        b=thickness(l,x,y,b)
    np.savetxt("weakcurvature.txt",curvaturedata)
    np.savetxt("weakxdata.txt",xdata)
    np.savetxt("weakydata.txt",ydata)
    np.savetxt("weakx1data.txt",x1data)
    np.savetxt("weaky1data.txt",y1data)
    np.savetxt("weakx2data.txt",x2data)
    np.savetxt("weaky2data.txt",y2data)
    np.savetxt("weakz.txt",zdata)    
    np.savetxt("weakddx.txt",ddxdata)
    np.savetxt("weakddy.txt",ddydata)


    return x,y,b
        
@jit(float64(float64[:],float64[:],float64[:]))
def plots(x,y,z):
    N=len(x)    

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

    return curvature
    

@jit(float64(float64[:],float64[:],float64[:],float64[:]))
def curve(x,y,z,b): 
    N=len(x)
    xmid=(x[0:N-1] + x[1:N])*0.5
    ymid=(y[0:N-1] + y[1:N])*0.5
    
    slope=np.zeros(N-1)    
    for i in range(N-1):
        if z[i]>0.5:
            slope[i]=(ymid[i]-ymid[i-1])/(xmid[i]-xmid[i-1])
        elif z[i]<0.5:
            slope[i]=(ymid[i+1]-ymid[i])/(xmid[i+1]-xmid[i])
    for i in range(N):
        if z[i]==0.5:
            slope[i]=(slope[i+1]+slope[i-1])*0.5
  
    x1=np.ones(N-1)
    y1=np.ones(N-1)
    x2=np.ones(N-1)
    y2=np.ones(N-1)
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
                 
    return x1,y1,x2,y2
    
    

    

        