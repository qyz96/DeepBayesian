#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:45:19 2019

@author: yizhouqian
"""

import numpy as np
 


## Compute the covariance matrix between different coordinates
def Cov(x, y, c_f):
    z=((x-x.T)**2+(y-y.T)**2)**(0.5)
    return c_f(z)

def CrossCov(x1,x2, y1, y2, c_f):
    z=((x1-x2)**2+(y1-y2)**2)**(0.5)
    return c_f(z)
    
def random_jump(x,y,c,rs, rl,m):
    k=np.random.uniform(-m,m)
    x0=np.random.uniform(0,c)
    y0=np.random.uniform(0,c)
    r1=np.random.uniform(rs,rl)**2
    r2=np.random.uniform(rs,rl)**2
    T1=np.abs(x-x0) < r1
    temp=((np.abs(x-x0) > r1) & (np.abs(x-x0) < r1+ 0.1))
    temp=temp.astype(float)
    temp*=1-np.abs(np.abs(x-x0) - r1)*10
    T1=T1*1+temp
    T2= np.abs(y-y0) < r2
    temp2=((np.abs(y-y0) > r2) & (np.abs(y-y0) < r2+ 0.1))
    temp2=temp2.astype(float)
    temp2*=1-np.abs(np.abs(y-y0)-r2)*10
    T2=T2*1+temp2
    
    return k*np.minimum(T1,T2),x0,y0


def random_jump_n(x,y,c,rs, rl,m):
    k=np.random.uniform(-m,m)
    x0=np.random.normal(0.9, 0.05)
    y0=np.random.normal(0.9, 0.05)
    x0=np.random.uniform(0,1)
    y0=np.random.uniform(0,1)
    r1=(rs+rl)/3
    r2=(rs+rl)/3
    T1=np.abs(x-x0) < r1
    temp=((np.abs(x-x0) > r1) & (np.abs(x-x0) < r1+ 0.00001))
    temp=temp.astype(float)
    temp*=1-np.abs(np.abs(x-x0) - r1)*10
    T1=T1*1+temp
    T2= np.abs(y-y0) < r2
    temp2=((np.abs(y-y0) > r2) & (np.abs(y-y0) < r2+ 0.00001))
    temp2=temp2.astype(float)
    temp2*=1-np.abs(np.abs(y-y0)-r2)*10
    T2=T2*1+temp2
    
    return k*np.minimum(T1,T2), x0, y0, k/m



def random_jump_v(x,y,c,rs, rl,m):
    k=np.random.uniform(-m,m)
    k=m/2
    x0=np.random.uniform(0,c)
    y0=np.random.uniform(0,c)
    r1=np.random.uniform(rs,rl)
    r2=np.random.uniform(rs,rl)
    r1=(rs+rl)/3
    r2=(rs+rl)/3
    T1=np.abs(x-x0) < r1
    temp=((np.abs(x-x0) > r1) & (np.abs(x-x0) < r1+ 0.00001))
    temp=temp.astype(float)
    temp*=1-np.abs(np.abs(x-x0) - r1)*10
    T1=T1*1+temp
    T2= np.abs(y-y0) < r2
    temp2=((np.abs(y-y0) > r2) & (np.abs(y-y0) < r2+ 0.00001))
    temp2=temp2.astype(float)
    temp2*=1-np.abs(np.abs(y-y0)-r2)*10
    T2=T2*1+temp2
    
    return k*np.minimum(T1,T2),x0,y0

def random_jump_v2(x,y,c,rs, rl,m):
    k=np.random.uniform(-m,m)
    x0=np.random.uniform(0,c)
    y0=np.random.uniform(0,c)
    r1=np.random.uniform(rs,rl)
    r2=np.random.uniform(rs,rl)
    T1=np.abs(x-x0) < r1
    temp=((np.abs(x-x0) > r1) & (np.abs(x-x0) < r1+ 0.00001))
    temp=temp.astype(float)
    temp*=1-np.abs(np.abs(x-x0) - r1)*10
    T1=T1*1+temp
    T2= np.abs(y-y0) < r2
    temp2=((np.abs(y-y0) > r2) & (np.abs(y-y0) < r2+ 0.00001))
    temp2=temp2.astype(float)
    temp2*=1-np.abs(np.abs(y-y0)-r2)*10
    T2=T2*1+temp2
    
    return k*np.minimum(T1,T2),x0,y0

def f_sum(a1,b1,b2, xx1, yy1, m1,m2, n2):
    z=np.zeros((m1,m2))
    for i in range(n2):
        bb=1e-4
        z+=np.power(bb,b1[i]+b2[i])*a1[i]*np.sin(b1[i]*np.pi*xx1)*np.sin(b2[i]*np.pi*yy1)\
        +np.power(bb,b1[i+n2]+b2[i+n2])*a1[i+n2]*np.sin(b1[i+n2]*np.pi*xx1)*np.cos(b2[i+n2]*np.pi*yy1)\
        +np.power(bb,b1[i+2*n2]+b2[i+2*n2])*a1[i+2*n2]*np.cos(b1[i+2*n2]*np.pi*xx1)*np.sin(b2[i+2*n2]*np.pi*yy1)\
        +np.power(bb,b1[i+3*n2]+b2[i+3*n2])*a1[i+3*n2]*np.cos(b1[i+3*n2]*np.pi*xx1)*np.cos(b2[i+3*n2]*np.pi*yy1)
    return z
        

def random_profile(xx1, yy1, xx2, yy2,m,freq):
    a1 = np.random.uniform(-1,1,m*4)
    b1 = np.random.randint(0,freq,size=m*4)
    b2 = np.random.randint(0,freq,size=m*4)
    vv=f_sum(a1,b1,b2,xx1,yy1,yy1.shape[0],xx1.shape[1],m)
    dd=f_sum(a1,b1,b2,xx2,yy2,yy2.shape[0],xx2.shape[1],m)
#    C=10/np.max(np.abs(dd))
#    vv*=C
#    dd*=C
    return vv, dd

def random_noise(low, high, shape, cov_l):
    n1,n2=shape
    noise=np.random.normal(scale=high,size=n1*n2)
    noise=cov_l.dot(noise)
    noise=np.reshape(noise,(n1,n2))
    return noise
    

def mesh_grid(num_x, num_y, ny_s, nx_s):
    gridx = np.arange(0,1, 1/num_x)
    gridy = np.arange(0,1, 1/num_y)
    gridxf = gridx[0:num_x:nx_s]
    gridyf = gridy[0:num_y:ny_s]

    xx1,yy1=np.meshgrid(gridxf,gridyf,sparse=True)
    xx2,yy2=np.meshgrid(gridx,gridy,sparse=True)
    return xx1, yy1, xx2, yy2

def random_transflip(A):
    B=np.flip(A,axis=1)
    if (np.random.randint(0,2)==3):
        B=np.flip(B,0)
    if (np.random.randint(0,11)<0):
        t=np.random.randint(0,(int)(A.shape[1]/2)+1)
        B=B[:,t:]
        B=np.insert(B,[B.shape[1]]*(t),B[:,[B.shape[1]-1]],axis=1)
        
    return np.flip(B,axis=1)



def data_generation(num_x, num_y, num_t, NumOfSamples, nx_s, ny_s, freq, river, low, high, cov_l):

    v=[]
    d=[]
    xx1,yy1,xx2,yy2=mesh_grid(num_x, num_y, ny_s, nx_s)
    
    for j in range(NumOfSamples):
        #print(j)
        vv, dd=random_profile(xx1, yy1, xx2, yy2, num_t, freq)
        noise=random_noise(low, high, river.shape, cov_l)
        x,y=np.meshgrid(xx2,yy2,sparse=False)
        jump,s1,s2=random_jump_v(x,y,1,0.15, 0.5, 24)
        dd+=jump+0*noise
        vv+=jump[0:num_y:ny_s, 0:num_x:nx_s]+0*noise[0:num_y:ny_s, 0:num_x:nx_s]
        landscape=random_transflip(river)
        river_d=landscape+0.15*noise
        river_v=river_d[0:num_y:ny_s, 0:num_x:nx_s]
        temp_v=(landscape+0.1*noise)[0:num_y:ny_s, 0:num_x:nx_s]+vv
        vv=Concat(vv, TakeAverage2(dd, num_y, num_x, ny_s, nx_s))
        river_v=Concat(river_v, TakeAverage2(river_d, num_y, num_x, ny_s, nx_s))
        temp_v=Concat(temp_v, TakeAverage2(landscape+dd+0.1*noise, num_y, num_x, ny_s, nx_s))
#        v.append(np.reshape(vv, (vv.shape[0]*vv.shape[1])))
#        v.append(np.reshape(temp_v, (vv.shape[0]*vv.shape[1])))
#        v.append(np.reshape(river_v, (vv.shape[0]*vv.shape[1])))
#        d.append(np.reshape(dd,(dd.shape[0], dd.shape[1],1)))
#        d.append(np.reshape(landscape+dd+0.1*noise,(dd.shape[0], dd.shape[1],1)))
#        d.append(np.reshape(river_d,(dd.shape[0], dd.shape[1],1)))
        v.append(np.reshape(vv,(vv.shape[0]*vv.shape[1])))
        v.append(np.reshape(temp_v, (vv.shape[0]*vv.shape[1])))
        v.append(np.reshape(river_v, (vv.shape[0]*vv.shape[1])))
        d.append(np.reshape(dd,(dd.shape[0]*dd.shape[1])))
        d.append(np.reshape(landscape+dd+0.1*noise,(dd.shape[0]*dd.shape[1])))
        d.append(np.reshape(river_d,(dd.shape[0]*dd.shape[1])))
        
    v=np.array(v)
    d=np.array(d)
    return v,d


def data_generation_gan(num_x, num_y, num_t, NumOfSamples, nx_s, ny_s, freq, river, low, high, cov_l):

    v=[]
    d=[]
    xx1,yy1,xx2,yy2=mesh_grid(num_x, num_y, ny_s, nx_s)
    
    for j in range(NumOfSamples):
        #print(j)
        vv, dd=random_profile(xx1, yy1, xx2, yy2, num_t, freq)
        noise=random_noise(low, high, river.shape, cov_l)
        x,y=np.meshgrid(xx2,yy2,sparse=False)
        jump,s1,s2=random_jump_v2(x,y,1,0.15, 0.5, 36)
        dd+=jump+0*noise
        vv+=jump[0:num_y:ny_s, 0:num_x:nx_s]+0*noise[0:num_y:ny_s, 0:num_x:nx_s]
        landscape=random_transflip(river)
        river_d=landscape+0.15*noise
        river_v=river_d[0:num_y:ny_s, 0:num_x:nx_s]
        temp_v=(landscape+0.1*noise)[0:num_y:ny_s, 0:num_x:nx_s]+vv
        vv=Concat(vv, TakeAverage2(dd, num_y, num_x, ny_s, nx_s))
        river_v=Concat(river_v, TakeAverage2(river_d, num_y, num_x, ny_s, nx_s))
        temp_v=Concat(temp_v, TakeAverage2(landscape+dd+0.1*noise, num_y, num_x, ny_s, nx_s))
#        v.append(np.reshape(vv, (vv.shape[0]*vv.shape[1])))
#        v.append(np.reshape(temp_v, (vv.shape[0]*vv.shape[1])))
#        v.append(np.reshape(river_v, (vv.shape[0]*vv.shape[1])))
#        d.append(np.reshape(dd,(dd.shape[0], dd.shape[1],1)))
#        d.append(np.reshape(landscape+dd+0.1*noise,(dd.shape[0], dd.shape[1],1)))
#        d.append(np.reshape(river_d,(dd.shape[0], dd.shape[1],1)))
        #v.append(np.reshape(vv,(vv.shape[0]*vv.shape[1])))
        v.append(np.reshape(temp_v, (vv.shape[0]*vv.shape[1])))
        v.append(np.reshape(river_v, (vv.shape[0]*vv.shape[1])))
        #d.append(np.reshape(dd,(dd.shape[0]*dd.shape[1])))
        d.append(np.reshape(landscape+dd+0.1*noise,(dd.shape[0]*dd.shape[1])))
        d.append(np.reshape(river_d,(dd.shape[0]*dd.shape[1])))
        
    v=np.array(v)
    d=np.array(d)
    return v,d

def Concat(vec1, vec2):
    return np.concatenate((np.reshape(vec1, (1,vec1.size)), np.reshape(vec2, (1, vec2.size))), axis=1)

def data_generation_n(num_x, num_y, num_t, NumOfSamples, ny_s, nx_s, freq, river, low, high, cov_l):

    v=[]
    d=[]
    xx1,yy1,xx2,yy2=mesh_grid(num_x, num_y, ny_s, nx_s)
    
    for j in range(NumOfSamples):
        #print(j)
        x,y=np.meshgrid(xx2,yy2,sparse=False)
        #noise=random_noise(low, high, river.shape, cov_l)
        jump, x0, y0, k=random_jump_n(x,y,1,0.15, 0.5, 32)
        vv=jump[0:num_y:ny_s, 0:num_x:nx_s]
        #vv=jump[0:num_y:n_s, 0:num_x:n_s]+noise[0:num_y:n_s, 0:num_x:n_s]
        v.append(np.reshape(vv, (vv.shape[0]*vv.shape[1])))
        #dd0=np.reshape(noise, (1,num_x*num_y))/5
        #dd=np.concatenate((np.array([x0,y0,k]), dd0), axis=None)
        #d.append(np.reshape(noise,(num_x*num_y)))
        d.append(np.reshape(jump, (jump.shape[0], jump.shape[1],1)))

        
    v=np.array(v)
    d=np.array(d)
    return v,d 


def TakeAverage(data, num_y, num_x, ny_s, nx_s):
    ry=num_y % ny_s
    rx=num_x % nx_s
    if (ry==0):
        ry=-num_y
    if (rx==0):
        rx=-num_x
    Temp=data[:-ry:ny_s,:-rx:nx_s]
    Ans=np.zeros(Temp.shape)
#    print(data.shape)
#    print(Temp.shape)
#    print(n_s)
    for i in range(ny_s):
        for j in range(nx_s):
            Ans+=data[i:-ry:ny_s, j:-rx:nx_s]
            
    Ans/=(ny_s*nx_s)
    return Ans


def TakeAverage2(data, num_y, num_x, ny_s, nx_s):
    ay=(int)((num_y+ny_s-1)/ny_s)
    ax=(int)((num_x+nx_s-1)/nx_s)
    Temp=np.zeros((ay,ax))
    Area=np.zeros((ay,ax))
    for i in range(num_y):
        for j in range(num_x):
            ii = (int)(i/ny_s)
            jj = (int)(j/nx_s)
            Temp[ii,jj]+=data[i,j]
            Area[ii,jj]+=1
            
    Ans = np.divide(Temp, Area)
#    print(data.shape)
#    print(Temp.shape)
#    print(n_s)
    return Ans



def kriging(x_coord, y_coord, data, ny_s, nx_s, c_f):
    n_y, n_x=x_coord.shape
    x1=x_coord[0:n_y:ny_s, 0:n_x:nx_s]
    y1=y_coord[0:n_y:ny_s, 0:n_x:nx_s]
    x_coord=np.reshape(x_coord, (n_y*n_x,1))
    y_coord=np.reshape(y_coord, (n_y*n_x,1))
    x1=np.reshape(x1, (x1.shape[0]*x1.shape[1],1))
    y1=np.reshape(y1, (y1.shape[0]*y1.shape[1],1))
    Qy0=c_f(((x1-x_coord.T)**2+(y1-y_coord.T)**2)**(0.5))
    Qyy=Cov(x1, y1, c_f)
    X=np.ones(x1.shape)
    X0=np.ones(x_coord.shape)
    MEGA=np.linalg.solve(np.block([[-Qyy, X],[X.T, 0]]), (np.block([[-Qy0],[X0.T]])))
    data_in=np.reshape(data,(x1.shape))
    data_out=MEGA[:-1,:].T.dot(data_in)

    
    
    return data_out, Qyy, Qy0


def cokriging(x_coord, y_coord, data, ny_s, nx_s, c_f1, c_f2, c_f3):
    n_y, n_x=x_coord.shape
    x1=x_coord[0:n_y:ny_s, 0:n_x:nx_s]
    y1=y_coord[0:n_y:ny_s, 0:n_x:nx_s]
    x2=x1[0:-1,0:-1]
    y2=y1[0:-1,0:-1]
    x_coord=np.reshape(x_coord, (n_y*n_x,1))
    y_coord=np.reshape(y_coord, (n_y*n_x,1))
    x1=np.reshape(x1, (x1.shape[0]*x1.shape[1],1))
    y1=np.reshape(y1, (y1.shape[0]*y1.shape[1],1))
    x2=np.reshape(x2, (x2.shape[0]*x2.shape[1],1))
    y2=np.reshape(y2, (y2.shape[0]*y2.shape[1],1)) 
    Q11=Cov(x1,y1,c_f1)
    Q12=CrossCov(x1,x2.T,y1,y2.T, c_f2)
    Q22=Cov(x2,y2,c_f3)
    Qyy=np.block([[Q11, Q12],[Q12.T, Q22]])
    Qy10=c_f1(((x1-x_coord.T)**2+(y1-y_coord.T)**2)**(0.5))
    Qy20=c_f2(((x2-x_coord.T)**2+(y2-y_coord.T)**2)**(0.5))
    X1=np.ones(x1.shape)
    X2=np.ones(x2.shape)
    X=np.block([[X1, np.zeros(X1.shape)],[np.zeros(X2.shape), X2]])
    Qy0=np.block([[Qy10], [Qy20]])
    X0=np.ones(x_coord.shape)
    X00=np.zeros(x_coord.shape)
    MEGA=np.linalg.solve(np.block([[-Qyy, X],[X.T, np.zeros((2,2))]]), (np.block([[-Qy0],[X0.T],[X00.T]])))
    data_in=np.reshape(data,(x1.shape[0]+x2.shape[0],1))
    data_out=MEGA[:-2,:].T.dot(data_in)

    return data_out, MEGA[:-2,:].T


def test(num_x, num_y, num_t, NumOfSamples, ny_s, nx_s, freq, low, high):

    v=[]
    d=[]
    loc=[]
    J=[]
    xx1,yy1,xx2,yy2=mesh_grid(num_x, num_y, ny_s, nx_s)
    
    for j in range(NumOfSamples):
        print(j)
        vv, dd=random_profile(xx1, yy1, xx2, yy2, num_t, freq)
        x,y=np.meshgrid(xx2,yy2,sparse=False)
        jump, x0, y0, r1, r2, k=random_jump(x,y,1,0.15, 0.5, 2)
        dd+=jump
        vv+=jump[0:num_y:ny_s, 0:num_x:nx_s]
        n1, n2=vv.shape
        vv=np.reshape(vv, (1,n1*n2))
        dd=np.reshape(dd, (1,num_x*num_y))
        AA=np.array([[x0]])
        v.append(vv)
        d.append(dd)
        loc.append(AA)
        J.append(jump)

        
    v=np.array(v)
    d=np.array(d)
    loc=np.array(loc)
    return v,d, loc, J

    
    
