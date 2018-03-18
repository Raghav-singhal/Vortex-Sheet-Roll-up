# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:25:23 2015

@author: raghavsinghal
"""

from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt    
def sbs(M,T,dt):
    X1=np.loadtxt("circlexdata.txt")
    Y1=np.loadtxt("circleydata.txt")
    
    X2=np.loadtxt("circlex1data.txt")
    Y2=np.loadtxt("circley1data.txt")

    X3=np.loadtxt("circlex2data.txt")
    Y3=np.loadtxt("circley2data.txt")
    N=np.loadtxt("circleNdata.txt")
    #curvature=np.loadtxt("D01curvature.txt")
#    z=np.loadtxt("weakz.txt")
#    
#    X4=np.loadtxt('D01xmaxdata.txt')
#    Y4=np.loadtxt('D01ymaxdata.txt')
#    X5=np.loadtxt('D01xmindata.txt')
#    Y5=np.loadtxt('D01ymindata.txt')
#     
#    X6=np.loadtxt('D01outxmaxdata.txt')
#    Y6=np.loadtxt('D01outymaxdata.txt')
#    X7=np.loadtxt('D01outxmindata.txt')
#    Y7=np.loadtxt('D01outymindata.txt')
#
#    X8=np.loadtxt('D01inxmaxdata.txt')
#    Y8=np.loadtxt('D01inymaxdata.txt')
#    X9=np.loadtxt('D01inxmindata.txt')
#    Y9=np.loadtxt('D01inymindata.txt')
    
    Finaltime=int(T/dt)
    #N=M
    fig = plt.figure(figsize=(20,10))

    ax = plt.axes(xlim=(-2, 2.5), ylim=(-1.3,2.3))

    line1, = ax.plot([], [],'ro',lw=3)
    line2, = ax.plot([], [],'ko',lw=3)
    line3, = ax.plot([], [],'ko',lw=3)
#    line4, = ax.plot([], [],'k',lw=3)
#    line5, = ax.plot([], [],'bs',lw=10)
#    line6, = ax.plot([], [],'bs',lw=10)
#    line7, = ax.plot([], [],'bs',lw=10)
#    line8, = ax.plot([], [],'bs',lw=10)
#    line9, = ax.plot([], [],'bs',lw=10)
#    line10, = ax.plot([], [],'bs',lw=10)


    time_template = 'time = %1f'
    time_text = ax.text(0.4, 0.9, '',fontsize=20, transform=ax.transAxes)
    N_text = ax.text(0.4, 0.87, '', fontsize=20,transform=ax.transAxes)

# initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
#        line4.set_data([], [])
#        line5.set_data([], [])
#        line6.set_data([], [])
#        line7.set_data([], [])
#        line8.set_data([], [])
#        line9.set_data([], [])
#        line10.set_data([], [])

        time_text.set_text('')
        N_text.set_text('')

        return line1, time_text, N_text,line2,line3
    
#animation function.  #This is called sequentially
    def animate(i):
        x1 = X1[np.sum(N[0:i]):np.sum(N[0:i+1])]
        y1 = Y1[np.sum(N[0:i]):np.sum(N[0:i+1])]
    
        x2 = X2[np.sum(N[0:i]):np.sum(N[0:i+1])]
        y2 = Y2[np.sum(N[0:i]):np.sum(N[0:i+1])]

        x3 = X3[np.sum(N[0:i]):np.sum(N[0:i+1])]
        y3 = Y3[np.sum(N[0:i]):np.sum(N[0:i+1])]
        
        line1.set_data(x1, y1)
        line2.set_data(x2, y2)
        line3.set_data(x3, y3)
        
        time_text.set_text(time_template%(i*dt))
        N_text.set_text('N = %0.00001f' % int(N[i]))

        return line1
#    def animate(i):
#        x1 = X1[(N)*i:(N)*(i+1)] 
#        y1 = Y1[(N)*i:(N)*(i+1)]
        
#        x2 = X2[(N)*i:(N)*(i+1)]
#        y2 = Y2[(N)*i:(N)*(i+1)]

#        x3 = X3[(N)*i:(N)*(i+1)]
#        y3 = Y3[(N)*i:(N)*(i+1)]
        
#        y4 = curvature[(N)*i:(N)*(i+1)]*0.5
#        x4 = z[(N)*i:(N)*(i+1)]

#        x5=X4[i] + 1.2
#        y5=Y4[i]*7
#
#        x6=X5[i] + 1.2
#        y6=Y5[i]*7        
#        
#        x7=X6[i] + 1.2
#        y7=Y6[i]*7        
#        
#        x8=X7[i] + 1.2
#        y8=Y7[i]*7        
#
#        x9=X8[i] + 1.2
#        y9=Y8[i]*7        
#
#        x10=X9[i] + 1.2
#        y10=Y9[i]*7        

         
#        line1.set_data(x1, y1)
#        line2.set_data(x2, y2)
#        line3.set_data(x3, y3)
#        line4.set_data(x4,y4)
#        line5.set_data(x5,y5)
#        line6.set_data(x6,y6)
#        line7.set_data(x7,y7)
#        line8.set_data(x8,y8)
#        line9.set_data(x9,y9)
#        line10.set_data(x10,y10)

        
#        time_text.set_text(time_template%(i*dt))
#        N_text.set_text('N =100 ')
#        return line1
        
# call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=Finaltime,blit=True)#, interval=1)
    anim.save('circleNEW.mp4', fps=1)#,dpi=200)
    plt.show()
