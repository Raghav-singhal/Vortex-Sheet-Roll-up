# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:57:44 2015

@author: raghavsinghal
"""
import numpy as np
import matplotlib.pyplot as plt

def F(X,e,a,b,g):
    
    ua=X[0,0]
    ub=X[1,0]
    ug=X[2,0]
    u_a=X[3,0]
    u_b=X[4,0]
    u_g=X[5,0]

    va=X[6,0]
    vb=X[7,0]
    vg=X[8,0]
    v_a=X[9,0]
    v_b=X[10,0]
    v_g=X[11,0]

    ha=X[12,0]
    hb=X[13,0]
    hg=X[14,0]
    h_a=X[15,0]
    h_b=X[16,0]
    h_g=X[17,0]
    
    dx=np.zeros((18,1)) + 1j*np.zeros((18,1))
    
    dx[0,0]=va - 1j*a*ha - e*(1j*a*ug*u_b)
    dx[1,0]=vb - 1j*b*hb - e*(1j*b*ug*u_a)
    dx[2,0]=vg - 1j*g*hg - e*(1j*g*ua*ub)
    
    dx[3,0]=-(ua + e*(1j*b*ug*v_b - 1j*g*vg*u_b))
    dx[4,0]=-(ub + e*(1j*a*ug*v_a - 1j*g*vg*ua))
    dx[5,0]=-(ug + e*(1j*a*va*ub + 1j*b*vb*ua))
    
    dx[6,0]=-(1j*a*ua + e*1j*a*(ug*h_b + u_b*hg))
    dx[7,0]=-(1j*b*ub + e*1j*b*(ug*h_a + u_a*hg))
    dx[8,0]=-(1j*g*ug + e*1j*g*(ua*hb + ub*ha))
    
    # complex conjugates
    
    dx[9,0]=v_a + 1j*a*h_a + e*(1j*a*ub*u_g)
    dx[10,0]=v_b + 1j*b*h_b + e*(1j*b*ua*u_g)
    dx[11,0]=v_g + 1j*g*h_g + e*(1j*g*u_a*u_b)
    
    dx[12,0]=-u_a + e*(1j*g*ub*v_g - 1j*a*vb*u_g)
    dx[13,0]=-u_b + e*(1j*g*ua*v_g - 1j*b*vb*u_g)
    dx[14,0]=-u_g + e*(1j*a*v_a*u_b + 1j*b*v_b*u_a)
    
    dx[15,0]=(1j*a*u_a + e*1j*a*(u_g*hb + ub*h_g))
    dx[16,0]=(1j*b*u_b + e*1j*b*(u_g*ha + ua*h_g))
    dx[17,0]=(1j*g*u_g + e*1j*g*(u_a*h_b + u_b*h_a))
    
    #print X,dx
    
    return dx
    
def RK4(X,T,e,a,b,g):
    dt=0.01
    Finaltime=int(T/dt)

    va=X[6]
    vb=X[7]
    vg=X[8]
    v_a=X[9]
    v_b=X[10]
    v_g=X[11]

    ha=X[12]
    hb=X[13]
    hg=X[14]
    h_a=X[15]
    h_b=X[16]
    h_g=X[17]
    
    qa=1j*a*va - ha
    qb=1j*b*vb - hb
    qg=1j*g*vg - hg

    q_a=-1j*a*v_a - h_a
    q_b=-1j*b*v_b - h_b
    q_g=-1j*g*v_g - h_g
    
    q1=np.abs(qa + q_a)
    q2=np.abs(qb + q_b)
    q3=np.abs(qg + q_g) 

    
    Q=q3
    T=np.ones(1)
    for t in range(Finaltime):
        T=np.append(T,t*dt)
        
        K1=dt*F(X,e,a,b,g)
        
        K2=dt*F(X + 0.5*K1,e,a,b,g)
        
        K3=dt*F(X + 0.5*K2,e,a,b,g)
        
        K4=dt*F(X + K3,e,a,b,g)
        
        X=X + (1./6.)*(K1 + 2*K2 + 2*K3 + K4)

        #X=X + dt*F(X,e,a,b,g)
        
        va=X[6]
        vb=X[7]
        vg=X[8]
        v_a=X[9]
        v_b=X[10]
        v_g=X[11]

        ha=X[12]
        hb=X[13]
        hg=X[14]
        h_a=X[15]
        h_b=X[16]
        h_g=X[17]
    
        qa=1j*a*va - ha
        qb=1j*b*vb - hb
        qg=1j*g*vg - hg

        q_a=-1j*a*v_a - h_a
        q_b=-1j*b*v_b - h_b
        q_g=-1j*g*v_g - h_g
    
        q1=np.abs(qa + q_a)
        q2=np.abs(qb + q_b)
        q3=np.abs(qg + q_g) 
        Q=np.append(Q,q3)
        
    return X,T,Q
def wavevortex(T,e):
    a=2
    b=3
    g=5
    
    ha=1 + 0*1j
    hb=3 + 0*1j
    hg=1 + 0*1j
    h_a=2 + 0*1j
    h_b=5 + 0*1j
    h_g=2 + 0*1j    
    
    ua=0 + 0*1j
    ub=0 + 0*1j
    ug=0 + 0*1j
    u_a=0 + 0*1j
    u_b=0 + 0*1j
    u_g=0 + 0*1j

    va=1j*a*ha
    vb=1j*b*hb
    vg=1j*g*hg
    v_a=-1j*a*h_a
    v_b=-1j*b*h_b
    v_g=-1j*g*h_g
    
    X=np.zeros((18,1)) + 1j*np.zeros((18,1))
    
    X[0]=ua
    X[1]=ub
    X[2]=ug
    X[3]=u_a
    X[4]=u_b
    X[5]=u_g
    
    X[6]=va
    X[7]=vb 
    X[8]=vg 
    X[9]=v_a    
    X[10]=v_b 
    X[11]=v_g
    
    X[12]=ha    
    X[13]=hb 
    X[14]=hg 
    X[15]=h_a 
    X[16]=h_b
    X[17]=h_g
    X,T,Q=RK4(X,T,e,a,b,g)
    
    plt.plot(T,Q)