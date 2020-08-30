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

def random_jump_v(x, y):
    k=12
    x0=np.random.uniform(0,1)
    y0=np.random.uniform(0,1)
    r=0.22
    t1=np.abs(x-x0) < r
    t2= np.abs(y-y0) < r
    return k*np.minimum(t1,t2),x0,y0

def mesh_grid(num_x, num_y, ny_s, nx_s):
    gridx = np.arange(0,1, 1/num_x)
    gridy = np.arange(0,1, 1/num_y)
    gridxf = gridx[0:num_x:nx_s]
    gridyf = gridy[0:num_y:ny_s]

    xx1,yy1=np.meshgrid(gridxf,gridyf,sparse=True)
    xx2,yy2=np.meshgrid(gridx,gridy,sparse=True)
    return xx1, yy1, xx2, yy2

def data_generation(num_x, num_y, num_sample, nx_s, ny_s, rivers, L, H):

    measurement=[]
    unknown=[]
    xx1,yy1,xx2,yy2=mesh_grid(num_x, num_y, ny_s, nx_s)
    for river in rivers:
        for j in range(num_sample):
            x,y=np.meshgrid(xx2,yy2,sparse=False)
            jump,_,_=random_jump_v(x,y)
            x_rg = river + np.reshape(L.dot(np.random.normal(0,1,size=(num_y * num_x,1))), (num_y, num_x))
            x_jp = river + np.reshape(L.dot(np.random.normal(0,1,size=(num_y * num_x,1))), (num_y, num_x)) + jump
            x_rg = x_rg.flatten()
            x_jp = x_jp.flatten()
            y_rg = H.dot(x_rg)
            y_jp = H.dot(x_jp)
            measurement.append(y_rg)
            measurement.append(y_jp)
            unknown.append(x_rg)
            unknown.append(x_jp)
        
    measurement=np.array(measurement)
    unknown=np.array(unknown)
    return measurement, unknown
