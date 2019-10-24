#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:53:38 2019

@author: yizhouqian
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import tensorflow as tf
from tensorflow import keras
import Data_Generation
import time



class Trainer2D:
    def __init__(self, x, y, t, sample, ny_s, nx_s, f, p, river):
        self.num_x=x
        self.num_y=y
        self.num_t=t
        self.num_sample=sample
        self.ny_s=ny_s
        self.nx_s=nx_s
        self.freq=f
        self.param=p

        self.river=river
        self.xx1,self.yy1,self.xx2,self.yy2=Data_Generation.mesh_grid(self.num_x, self.num_y, self.ny_s, self.nx_s)
        self.x_coord, self.y_coord=np.meshgrid(self.xx2,self.yy2, sparse=False)
        self.x_coord=np.reshape(self.x_coord, (self.num_y*self.num_x,1))
        self.y_coord=np.reshape(self.y_coord, (self.num_y*self.num_x,1))
        self.covar= lambda x : p['psill']*np.exp(-(x**2)/(p['range']**2))
        self.Cov=Data_Generation.Cov(self.x_coord, self.y_coord, self.covar)
        self.model=dict()
        
        return

    #Cov_L=np.linalg.cholesky(Cov)


    def GenerateData(self, number):
        low=0
        high=0.5
        for i, river in enumerate(self.river):
            print(i)
            if (i==0):
                v, d=Data_Generation.data_generation(self.num_x, self.num_y, self.num_t,self.num_sample,self.ny_s, self.nx_s,self.freq, river, low, high, self.Cov)
            else:
                vv,dd=Data_Generation.data_generation(self.num_x, self.num_y, self.num_t,self.num_sample,self.ny_s, self.nx_s,self.freq, river, low, high, self.Cov)
                v=np.vstack([v, vv])
                d=np.vstack([d,dd])
        np.save("Data/traininginput"+str(number), v)
        np.save("Data/trainingoutput"+str(number), d)
        return v,d

        
    def LoadData(self, number, noiselevel):
        self.v=np.load("Data/traininginput"+str(number)+".npy")
        self.d=np.load("Data/trainingoutput"+str(number)+".npy")

#        Noise=np.random.normal(0,noiselevel, size=self.v.shape)
#        self.v+=Noise
#        
#        n1,n2,n3=self.d.shape
#        m1,m2,m3=self.v.shape
#        self.d=np.reshape(self.d, (n1,n2*n3))
#        self.v=np.reshape(self.v,(m1,m2*m3))
        return self.v, self.d


    def CreateNetwork(self, Hidden_layer, fname, Regularizer, name):

        self.T=int(self.v.shape[0]*9/10)
        self.T=((int)(self.T/32))*32
        Model=keras.Sequential()
        for i in range(len(Hidden_layer)):
            Model.add(keras.layers.Dense(Hidden_layer[i], kernel_initializer='random_uniform',kernel_regularizer=Regularizer, activation=fname))
        Model.add(keras.layers.Dense(self.d.shape[1], activation='linear'))
        self.model[name]=Model
        return

    def TrainNetwork(self, Optimizer, ep, name, los='mse'):
        self.model[name].compile(optimizer=Optimizer, loss=los, metrics=['mae'])
        self.model[name].fit(self.v[:self.T,:], self.d[:self.T,:], epochs=ep, batch_size=16, validation_data=(self.v[self.T:,:], self.d[self.T:,:]))
        return

    #%%





####Generate Comparisons
    def compare(self, covar1, covar2, noiselevel, itn, river):

        #ind=np.random.randint(0,self.num_sample*3)
        x_coord, y_coord=np.meshgrid(self.xx2,self.yy2, sparse=False)
        dd=np.reshape(river,(self.num_y,self.num_x))
        print(dd.shape)
        vv=self.Sparse(dd)

        v1=0
        v2=7

        self.v_noise=np.random.normal(0,noiselevel,vv.shape)
        vv=dd[0:self.num_y:self.ny_s, 0:self.num_x:self.nx_s]+self.v_noise
        data_out=Data_Generation.kriging(x_coord, y_coord, vv, self.ny_s, self.nx_s, covar2)
        data_out=np.reshape(data_out,(self.num_y, self.num_x))
        vv_d=self.Model_d.predict(np.reshape(vv,(1,self.yy1.shape[0]*self.xx1.shape[1])))
        vv_d=np.reshape(vv_d,(self.yy1.shape[0],self.xx1.shape[1]))
        dl=[]
        name=[]
        dlr0=[]
        #print(river.shape)
        for nn, model in self.model.items():
            dlt=self.DNNPredict(vv, nn)
            dl.append(dlt)
            #dlr, dlr0t=self.IterativeNNRK(dd,dl[0], itn, vv, x_coord, y_coord, covar1, nn)
            #print(nn)
            #dlr0t=np.zeros((self.num_y, self.num_x))
            name.append(nn)
            dlr0t=Data_Generation.kriging(x_coord, y_coord, vv_d-self.Sparse(dlt), self.ny_s,self.nx_s, covar2)
            dlr0t=np.reshape(dlr0t, (self.num_y, self.num_x))
            dlr0t+=dlt
            dlr0.append(dlr0t)
             
        ####  Denoise input

        ####
        



        return data_out, dl, dlr0, dd, name
    
    
    def DNNPredict(self,vv, nn):
        #vv=Data_Generation.TakeAverage(dd, self.num_y,self.num_x, self.ny_s, self.nx_s)
        #vv=self.Sparse(dd)
        vv_d=self.model[nn].predict(np.reshape(vv,(1,vv.shape[0]*vv.shape[1])))
        vv_d=np.reshape(vv_d,(self.num_y,self.num_x))
        
        return vv_d
#ok = Data_Generation.kriging(x_coord, y_coord, np.reshape(vt, vt.shape[0]*vt.shape[1]), param, n_s)

    def PlotandCompare(self, d, dd, n, norm, filename, Save=False, Error=True, ind=0, v1=0, v2=7):
        if (norm=='mse'):
            f=lambda x,y: np.mean((x-y)**2)**(0.5)
        else:
            f=lambda x,y: np.mean(np.abs(x-y))
        err=np.zeros(4)
        fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(9,9))
        for i in range(4):
            err[i]=f(d[i],dd)
            s=n[i]
            if (Error):
                s+=": "+norm+"= "+np.format_float_scientific(err[i],precision=2, exp_digits=1)
            im=axeslist.ravel()[i].imshow(-d[i], vmin=-v2-2, vmax=-v1)
            axeslist.ravel()[i].set_title(s)
            axeslist.ravel()[i].set_axis_off()
        cb_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cb_ax)
        plt.subplots_adjust(hspace=0.05, wspace=0.1)
        if (Save):
            plt.savefig(filename+".pdf", dpi=300)
        plt.show()
        
        return
        
        
        


    def IterativeNNRK(self, dd, dl,num, vv, x_coord, y_coord, covar1, name):
        dlrr=dl+0
        for i in range(num):
            rr_ok=Data_Generation.kriging(x_coord, y_coord, vv-dlrr[0:self.num_y:self.ny_s,0:self.num_x:self.nx_s], self.ny_s, self.nx_s, covar1)
            rr_ok=np.reshape(rr_ok, (self.num_y,self.num_x))
            dlrr+=rr_ok
            print(np.mean((dd-dlrr)**2)**(0.5))
            dlrr1=dlrr+0
            rr_dl=self.model[name].predict(np.reshape(vv-dlrr[0:self.num_y:self.ny_s,0:self.num_x:self.nx_s],(1,self.yy1.shape[0]*self.xx1.shape[1])))
            rr_dl=np.reshape(rr_dl, (self.num_y,self.num_x))
            dlrr+=rr_dl
            print(np.mean((dd-dlrr)**2)**(0.5))

        return dlrr, dlrr1
    
    def DenoisingNetwork(self, noiselevel, Hidden_layer, fname, Regularizer):
        self.v_n=self.v+np.random.normal(0,noiselevel, size=self.v.shape)
        self.Model_d=keras.Sequential()
        for i in range(len(Hidden_layer)):
            self.Model_d.add(keras.layers.Dense(Hidden_layer[i], kernel_initializer='random_uniform',kernel_regularizer=keras.regularizers.l2(Regularizer), activation=fname))
        self.Model_d.add(keras.layers.Dense(self.v.shape[1], activation='linear'))
        
        return
    
    def TrainDenoisingNetwork(self, Optimizer, ep):
        self.Model_d.compile(optimizer=Optimizer, loss='mse', metrics=['mae'])
        self.Model_d.fit(self.v_n[:self.T,:], self.v[:self.T,:], epochs=ep, batch_size=32, validation_data=(self.v_n[self.T:,:], self.v[self.T:,:]))
        return
    
    def Sparse(self, data):
        return data[0:self.num_y:self.ny_s, 0:self.num_x:self.nx_s]
    
    
    def ResidualTrainingData(self, noiselevel):
        vd=np.reshape(self.river, (1, self.num_y*self.num_x))
        vd=np.tile(vd,(self.num_sample, 1))
        for i in range(self.num_sample-1):
            vd[i,:]=np.reshape(Data_Generation.random_transflip(self.river), (1, self.num_y*self.num_x))
        vd_s=np.reshape(vd, (self.num_sample, self.num_y,self.num_x))[:,0:self.num_y:self.ny_s, 0:self.num_x:self.nx_s]
        vd_s=np.reshape(vd_s, (self.num_sample, self.yy1.shape[0]*self.xx1.shape[1]))
        vd_s+=np.random.normal(0,noiselevel,size=vd_s.shape)
        self.v_p=vd-self.Model.predict(vd_s)
        
    
    def ResidualNetwork(self, Hidden_layer, fname, Regularizer):
        self.v_s=np.reshape(self.v_p, (self.num_sample, self.num_y,self.num_x))[:,0:self.num_y:self.ny_s, 0:self.num_x:self.nx_s]
        self.v_s=np.reshape(self.v_s, (self.num_sample, self.yy1.shape[0]*self.xx1.shape[1]))
        self.Model_r=keras.Sequential()
        for i in range(len(Hidden_layer)):
            self.Model_r.add(keras.layers.Dense(Hidden_layer[i], kernel_initializer='random_uniform',kernel_regularizer=keras.regularizers.l2(Regularizer), activation=fname))
        self.Model_r.add(keras.layers.Dense(self.d.shape[1], activation='linear'))
        return
    
    def TrainResidualNetwork(self, Optimizer, ep):
        self.Model_r.compile(optimizer=Optimizer, loss='mse', metrics=['mae'])
        self.Model_r.fit(self.v_s[:self.T,:], self.v_p[:self.T,:], epochs=ep, batch_size=32, validation_data=(self.v_s[self.T:,:], self.v_p[self.T:,:]))
        return
    
    def ResidualPredict(self, ipt):
        opt=self.Model_r.predict(np.reshape(ipt,(1,self.yy1.shape[0]*self.xx1.shape[1])))
        opt=np.reshape(opt,(self.num_y,self.num_x))
        return opt
        
        
    
class RectangleTrainer(Trainer2D):
        def GenerateData(self, number):
            low=0
            high=0.5
            for i, river in enumerate(self.river):
                print(i)
                if (i==0):
                    v, d=Data_Generation.data_generation_n(self.num_x, self.num_y, self.num_t,self.num_sample,self.ny_s, self.nx_s, self.freq, river, low, high, self.Cov)
                else:
                    vv,dd=Data_Generation.data_generation_n(self.num_x, self.num_y, self.num_t,self.num_sample,self.ny_s, self.nx_s, self.freq, river, low, high, self.Cov)
                    v=np.vstack([v, vv])
                    d=np.vstack([d,dd])
            np.save("Data/traininginput"+str(number), v)
            np.save("Data/trainingoutput"+str(number), d)
            return v,d
        
        def LoadData(self, number, noiselevel):
            self.v=np.load("Data/traininginput"+str(number)+".npy")
            self.d=np.load("Data/trainingoutput"+str(number)+".npy")
    
            Noise=np.random.normal(0,noiselevel, size=self.v.shape)
            self.v+=Noise
            print(self.d.shape)
            n1,n2=self.d.shape
            m1,m2,m3=self.v.shape
            self.d=np.reshape(self.d, (n1,n2))
            self.v=np.reshape(self.v,(m1,m2*m3))
            return self.v, self.d
    
    
        def compare(self, covar1, covar2, noiselevel, itn, river):

        #ind=np.random.randint(0,self.num_sample*3)
            x_coord, y_coord=np.meshgrid(self.xx2,self.yy2, sparse=False)
            dd=np.reshape(river,(self.num_y,self.num_x))
            vv=self.Sparse(dd)
    
            v1=0
            v2=7
    
            self.v_noise=np.random.normal(0,noiselevel,vv.shape)
            vv=dd[0:self.num_y:self.ny_s, 0:self.num_x:self.nx_s]+self.v_noise
            data_out=Data_Generation.kriging(x_coord, y_coord, vv, self.ny_s, self.nx_s, covar2)
            data_out=np.reshape(data_out,(self.num_y, self.num_x))

                 
            ####  Denoise input
    
            ####
            
    
    
    
            return data_out

