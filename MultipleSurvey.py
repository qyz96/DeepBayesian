#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:37:01 2019

@author: yizhouqian
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from Training2D import Trainer2D
import tensorflow as tf
from tensorflow import keras
from Data_Generation import *
import keras.backend as K
#%%
m=51
n=75
def ReadFile(fname):
    data=[]
    with open(fname) as fp:
        for i, line in enumerate(fp):
            data.append(line[:-1].split(','))
    data=np.array(data)
    data=data.astype(float)
    if (data.shape[0]==3876):
        Ind= ((np.arange(data.shape[0]) % 76) != 0)
        data=data[Ind, :]
    return data[:,2], data.shape[0], data[:, 3:]


def WriteFile(fname, data):
    with open(fname, 'w') as fp:
        for depth in data:
            fp.write(np.format_float_scientific(depth,precision=4, exp_digits=1))
            fp.write('\n')
            
    
    return 


    

#%%
path='SurveyFull/'
depth=[]
bathytime=[]
i=0
for filename in glob.glob(os.path.join(path, '*.txt')):
    i+=1
    #print(filename)
    try:
        dd, ll, xy=ReadFile(filename)
    except:
        print(filename)
    if (ll == 3825):
        print(len(depth))
        print(i)
        dd=np.reshape(dd, (m,n))
        #dd=-np.flip(dd)
        dd=np.reshape(dd, m*n)
        depth.append(dd)
        bathytime.append(filename)
    
depth=np.array(depth)
#depth=np.vstack(depth)


#%%


for i in range(271):
    WriteFile('Depths/'+str(i)+'.txt', depth[i,:])
    
#%%


i=2
    
AA=np.loadtxt('Speed/Speed'+str(i)+'.txt')
plt.imshow(np.flip(np.reshape(AA,(m,n))))
plt.colorbar()
plt.show()
plt.imshow(np.flip(np.reshape(depth[i,:],(m,n))))
plt.colorbar()
plt.show()
#%%

SampleBathy=[]
SampleTime=[]
for ii in range(4):
    for jj in range(depth.shape[0]):
        #if (bathytime[jj][15:19]=='2000'):
#        SampleBathy.append(np.reshape(depth[jj],(m,n)))
#        SampleTime.append(bathytime[jj][15:23]) 
        print(bathytime[jj][15:23])
               


    
#Tn.PlotandCompare(SampleBathy, 0, SampleTime, 'mse', 'Sample', True, False, 0, -10, 5)



#%%
num_x=48
num_y=32
num_t=3
num_sample=60
ny_s=4
nx_s=4
freq=40

nug=0.4
r=0.07
p=1
param={'psill':p, 'range': r, 'nugget':nug}

river=[]
for i in range(239):
    rr=np.reshape(depth[i,:num_y*num_x], (num_y, num_x))
    river.append(rr)

Tn=Trainer2D(num_x, num_y, num_t, num_sample, ny_s, nx_s, freq, param, river)
#%%
##  101 normal data
##  102 changed gaussian covariance parameter, adjusted the magnitude of random fourier series
##  increased number of samples from 20 to 271
##  103 240 circles 
##  104 240 7
##  105 test diffusion
##  106 diffusion with different number of samples
##  108 Retest of 102
##  109 test
Tn.GenerateData(109)
#%%
def totalvariation(y_true, y_pred, num_x, num_y, ll, bs):
    y_temp=K.reshape(y_pred, (bs, num_y, num_x))
    tv=K.sum(K.abs(y_temp[:,2:,1:-1]-y_temp[:,1:-1,1:-1]))+\
    K.sum(K.abs(y_temp[:,0:-2,1:-1]-y_temp[:,1:-1,1:-1]))+\
    K.sum(K.abs(y_temp[:,1:-1,0:-2]-y_temp[:,1:-1,1:-1]))+\
    K.sum(K.abs(y_temp[:,1:-1,2:]-y_temp[:,1:-1,1:-1]))
    tv/=num_y*num_x
    return K.mean(K.abs(y_true-y_pred)**2)+ll*tv


def tvloss(num_x, num_y, ll, bs):
    def tvlos(y_true, y_pred):
        return totalvariation(y_true, y_pred, num_x, num_y, ll, bs)
    return tvlos


def l0tv(y_true, y_pred, num_x, num_y, ll, bs, thres):
    y_temp=K.reshape(y_pred, (bs, num_y, num_x))
#    tv=K.sum(K.cast(K.abs(y_temp[:,2:,1:-1]-y_temp[:,1:-1,1:-1])>thres, 'float32'))+\
#    K.sum(K.cast(K.abs(y_temp[:,0:-2,1:-1]-y_temp[:,1:-1,1:-1])>thres, 'float32'))+\
#    K.sum(K.cast(K.abs(y_temp[:,1:-1,0:-2]-y_temp[:,1:-1,1:-1])>thres, 'float32'))+\
#    K.sum(K.cast(K.abs(y_temp[:,1:-1,2:]-y_temp[:,1:-1,1:-1])>thres, 'float32'))
    tv=K.sum(K.sigmoid(K.abs(y_temp[:,2:,1:-1]-y_temp[:,1:-1,1:-1])-thres))+\
    K.sum(K.sigmoid(K.abs(y_temp[:,0:-2,1:-1]-y_temp[:,1:-1,1:-1])-thres))+\
    K.sum(K.sigmoid(K.abs(y_temp[:,1:-1,0:-2]-y_temp[:,1:-1,1:-1])-thres))+\
    K.sum(K.sigmoid(K.abs(y_temp[:,1:-1,2:]-y_temp[:,1:-1,1:-1])-thres))
    return K.mean(K.abs(y_true-y_pred)**2)+ll*tv


def tvl0loss(num_x, num_y, ll, bs, thres):
    def tvl0los(y_true, y_pred):
        return l0tv(y_true, y_pred, num_x, num_y, ll, bs, thres)
    return tvl0los


    
    
#%%


Tn.LoadData(109,0.2)

Hidden_layer=[1000]
fname='linear'
Optimizer = tf.train.AdamOptimizer(0.001)
##Optmizer = keras.optimizers.SGD(0.001, decay=1e-1)
Regularizer=0
Reg=keras.regularizers.l1(Regularizer)
los1=tvl0loss(Tn.num_x, Tn.num_y, 1e-1,4,2.5)
los2=tvloss(Tn.num_x, Tn.num_y, 1e-2,16)
los3='mae'


#%%
los1=tvl0loss(Tn.num_x, Tn.num_y, 1e-3,16,3.5)
NetworkName='L2+L0variation-Thres=3.5'
Tn.CreateNetwork(Hidden_layer, fname, Reg, NetworkName)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.001), 2, NetworkName,los1)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.0001), 2, NetworkName,los1)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.00001), 2, NetworkName, los1)



los1=tvl0loss(Tn.num_x, Tn.num_y, 1e-3,16,5.5)
NetworkName='L2+L0variation-Thres=5.5'
Tn.CreateNetwork(Hidden_layer, fname, Reg, NetworkName)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.001), 2, NetworkName,los1)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.0001), 2, NetworkName,los1)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.00001), 2, NetworkName, los1)


los1=tvl0loss(Tn.num_x, Tn.num_y, 1e-3,16,7.5)
NetworkName='L2+L0variation-Thres=7.5'
Tn.CreateNetwork(Hidden_layer, fname, Reg, NetworkName)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.001), 2, NetworkName,los1)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.0001), 2, NetworkName,los1)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.00001), 2, NetworkName, los1)

#%%
los2=tvloss(Tn.num_x, Tn.num_y, 1e-1,16)
NetworkName='L2+L1variation1e-0'
Tn.CreateNetwork(Hidden_layer, fname, Reg, NetworkName)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.001), 2, NetworkName,los2)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.0001), 2, NetworkName,los2)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.00001), 2, NetworkName, los2)


los2=tvloss(Tn.num_x, Tn.num_y, 5e-2,16)
NetworkName='L2+L1variation1e-1'
Tn.CreateNetwork(Hidden_layer, fname, Reg, NetworkName)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.001), 2, NetworkName,los2)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.0001), 2, NetworkName,los2)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.00001), 2, NetworkName, los2)


los2=tvloss(Tn.num_x, Tn.num_y, 1e-2,16)
NetworkName='L2+L1variation1e-2'
Tn.CreateNetwork(Hidden_layer, fname, Reg, NetworkName)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.001), 2, NetworkName,los2)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.0001), 2, NetworkName,los2)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.00001), 2, NetworkName, los2)

#%%

Tn.CreateNetwork(Hidden_layer, fname, Reg, 'L1')
Tn.TrainNetwork(tf.train.AdamOptimizer(0.001), 2, 'L1',los3)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.0001), 2, 'L1',los3)
Tn.TrainNetwork(tf.train.AdamOptimizer(0.00001), 2, 'L1', los3)

#%%


Hidden_layer2=[1000]
Optimizer = tf.train.AdamOptimizer(0.001)
gname='relu'
##Optmizer = keras.optimizers.SGD(0.001, decay=1e-1)
Regularizer2=0.01
Tn.DenoisingNetwork(0.5,Hidden_layer2, gname, Regularizer2)
Tn.TrainDenoisingNetwork(tf.train.AdamOptimizer(0.001), 8)
Tn.TrainDenoisingNetwork(tf.train.AdamOptimizer(0.0001), 2)
Tn.TrainDenoisingNetwork(tf.train.AdamOptimizer(0.00001), 2)

#%%
#from Training2D import Trainer2D
nn=0.1
r=0.4
p=0.2
s=1
covar0= lambda x : p*np.exp(-(x**2)/(r**2))+nn
covar1= lambda x : p*np.exp(-(x)/(r))+nn
covar2= lambda x : p+s*x
#ind=np.random.randint(0,4000)
ind=np.random.randint(0,270)
#ddt=np.reshape(Tn.d[3*ind+1,:],(m,n))
#ind=0
#rr=-np.copy(np.reshape(Tn.d[ind,:], (m,n)))
rr=(np.copy(np.reshape(depth[ind,:], (m,n))))
x,y=np.meshgrid(Tn.xx2,Tn.yy2,sparse=False)
jump,x0, y0=random_jump_v(x,y,1,0.2, 0.5, 32)
jump=np.abs(jump)
rr+=jump
#rr+=jump0
dok, dl, dlr,dd, name=Tn.compare(covar2, covar2, 0.2,1, rr)
D=[dok,dl[0],dlr[0],dd]
Residual=[dok-dd, dl-dd, dlr-dd, dd-dd]
dl.append(dd)
name.append('groundtruth')
#Tn.PlotandCompare(dl, dd,name, 'mse', 'L1variationcoeff', False, True, 0, -10, 5)
name2=['OK', 'DNN', 'NNRK', 'Groundtruth']
Tn.PlotandCompare(D, dd,name2, 'mse', 'Average2', False, True, 0, -10, 5)
#Tn.PlotandCompare(Residual, 0,name, 'mse', 'MCComparisonWJ', False, 0, -0.5, 0.5)

#%%

plt.imshow(dl[0]-dd)
plt.colorbar()
plt.show()


#%%

plt.plot(np.squeeze(Tn.xx2), dok[np.argmin(np.abs(Tn.yy2-y0)),:], label="OK")
#plt.plot(np.squeeze(Tn.xx2), dl[np.argmin(np.abs(Tn.yy2-y0)),:], label="L2")
#plt.plot(np.squeeze(Tn.xx2), dd[np.argmin(np.abs(Tn.yy2-y0)),:], label="groundtruth")
for i in range(len(name)):
    #if ((i==0) or (i==1) or (i==4)):
    plt.plot(np.squeeze(Tn.xx2), dl[i][np.argmin(np.abs(Tn.yy2-y0)),:],label=name[i])
#plt.plot(np.squeeze(Tn.xx2), dlr[np.argmin(np.abs(Tn.yy2-y0)),:],label="nnrk")
plt.scatter(np.squeeze(Tn.xx2)[::nx_s], dl[i][np.argmin(np.abs(Tn.yy2-y0)),::nx_s])
plt.legend()
plt.title("across-shore section")
#plt.savefig("Graphs/Average2cross-shore.pdf")
plt.show()
plt.close()

plt.plot(np.squeeze(Tn.yy2), dok[:,np.argmin(np.abs(Tn.xx2-x0))], label="OK")
#plt.plot(np.squeeze(Tn.yy2), dl[n:,np.argmin(np.abs(Tn.xx2-x0))], label="L2")
#plt.plot(np.squeeze(Tn.yy2), dd[:,np.argmin(np.abs(Tn.xx2-x0))], label="groundtruth")
for i in range(len(name)):
    #if ((i==0) or (i==1) or (i==4)):
    plt.plot(np.squeeze(Tn.yy2), dl[i][:,np.argmin(np.abs(Tn.xx2-x0))],label=name[i])
#plt.plot(np.squeeze(Tn.yy2), dlr[:,np.argmin(np.abs(Tn.xx2-x0))],label="nnrk")
plt.scatter(np.squeeze(Tn.yy2)[::ny_s], dl[i][::ny_s,np.argmin(np.abs(Tn.xx2-x0))])
plt.legend()
plt.title("along-shore section")
#plt.savefig("Graphs/Average2along-shore.pdf")
plt.show()

#%%

jump,x0, y0=random_jump(x,y,1,0.1, 0.2, 32)
plt.imshow(jump)
plt.title('x_gamma (Random Jumps)')
plt.savefig("Survey/BeamerPlots/Xgamma.pdf")
plt.show()


Noise=random_noise(0,5, (m,n), Tn.Cov)
plt.imshow(Noise)
plt.title('x_beta (Gaussian Noise)')
plt.savefig("Survey/BeamerPlots/Xbeta.pdf")
plt.show()


Profile=np.flip(np.copy(np.reshape(depth[251,:], (m,n))))
plt.imshow(Profile)
plt.title('x_alpha (Bathymetry Profile)')
plt.savefig("Survey/BeamerPlots/Xalpha.pdf")
plt.show()



#%%

rok=np.zeros(len(np.arange(240,271)))
rdl=np.zeros(len(np.arange(240,271)))
rnnrk=np.zeros(len(np.arange(240,271)))
for j,i in enumerate(np.arange(240,271)):
    rr=np.copy(np.reshape(depth[i,:], (m,n)))
    dok, dl, dlr,dd, name=Tn.compare(covar2, covar2, 0.2,1, rr)
    rok[j]=np.mean(np.abs(dd-dok)**2)**(0.5)
    rdl[j]=np.mean(np.abs(dd-dl)**2)**(0.5)
    rnnrk[j]=np.mean(np.abs(dd-dlr)**2)**(0.5)
    
    
#%%


plt.plot(np.arange(240,271), np.divide(rok-rdl, rok), label='dl')
plt.plot(np.arange(240,271), np.divide(rok-rnnrk, rok), label='nnrk')
plt.legend()

plt.plot()
PIdl=np.divide(rok-rdl, rok)
PInnrk=np.divide(rok-rnnrk, rok)
print(np.mean(rok))
print(np.mean(rdl))
print(np.mean(rnnrk))
print(np.mean(PIdl))
print(np.mean(PInnrk))

#%%

XX=np.array([704,2176, 3616, 5088, 10912, 18208, 36448, 72896, 109344, 145928])
YY=np.array([0.8647, 0.4464, 0.2917, 0.2378, 0.1638, 0.1482, 0.1391, 0.1374, 0.1371, 0.1373])
plt.scatter(XX,YY)
plt.plot(XX,YY)
plt.xlabel('Number of Training Samples')
plt.ylabel('Mean Squared Error')
plt.title('Convergence of Loss Function')
#plt.savefig('Graphs/Convergencevstrain.pdf')



#%%

NoiseStep=np.arange(0,0.5, 0.01)
Error=np.zeros((50,3))
for j in range(50):
    Err_t=np.zeros((100,3))
    print(j)
    for i in range(10):
        ind=np.random.randint(240,270)
        dok, dl, dlr,dd, name=Tn.compare(covar2, covar2, NoiseStep[j],1, rr)
        Err_t[i,0]=np.mean(np.abs(dd-dok)**2)**(0.5)
        Err_t[i,1]=np.mean(np.abs(dd-dl)**2)**(0.5)
        Err_t[i,2]=np.mean(np.abs(dd-dlr)**2)**(0.5)
    Error[j,:]=np.mean(Err_t, axis=0)
    
plt.close()
plt.plot(NoiseStep, Error[:,0], label="OK")
plt.plot(NoiseStep, Error[:,1], label="DNN")
#plt.plot(NoiseStep, Error[:,2], label="be2")
plt.plot(NoiseStep, Error[:,2], label="NNRK")
plt.legend()
#plt.title("Sensitivity to Noise")
plt.xlabel("Noise Level")
plt.ylabel("RMSE")
#plt.savefig('MSNoisePaper.pdf')
plt.show()





#%%
jj=np.random.randint(0, 42)
plt.imshow(np.reshape(-depth[ind,:], (m,n)), origin='lower')
#plt.colorbar()
plt.savefig('SampleUnknown.pdf')
plt.close()


plt.imshow(Tn.Sparse(np.reshape(-depth[ind,:], (m,n))), origin='lower')
#plt.colorbar()
plt.savefig('SampleMeasurement.pdf')
#plt.show()
#%%
ind=np.random.randint(0,10000)
ddt=np.reshape(Tn.d[3*ind+2,:],(m,n))
plt.imshow(np.reshape(ddt, (m,n)), origin='lower')
plt.colorbar()
plt.show()

