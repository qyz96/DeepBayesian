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
    def __init__(self, x, y, sample, ny_s, nx_s, r, rivers, H):
        self.num_x=x
        self.num_y=y
        self.num_sample=sample
        self.ny_s=ny_s
        self.nx_s=nx_s
        self.rivers=rivers
        self.xx1,self.yy1,self.xx2,self.yy2=Data_Generation.mesh_grid(self.num_x, self.num_y, self.ny_s, self.nx_s)
        self.x_coord, self.y_coord=np.meshgrid(self.xx2,self.yy2, sparse=False)
        self.x_coord=np.reshape(self.x_coord, (-1, 1))
        self.y_coord=np.reshape(self.y_coord, (-1, 1))
        self.covar= lambda x : np.exp(-(x**2)/(r**2))
        self.cov=0.15 * Data_Generation.Cov(self.x_coord, self.y_coord, self.covar)
        self.L = np.linalg.cholesky(self.cov + 0.000001 * np.identity(self.num_y * self.num_x))
        self.model=dict()
        self.H = H
        return

    def GenerateData(self, number):
        y, x=Data_Generation.data_generation(self.num_x, self.num_y, self.num_sample, self.ny_s, self.nx_s, self.rivers, self.L, self.H)
        np.save("Data/traininginput"+str(number), y)
        np.save("Data/trainingoutput"+str(number), x)
        return 
        
    def LoadData(self, number):
        self.y=np.load("Data/traininginput"+str(number)+".npy")
        self.x=np.load("Data/trainingoutput"+str(number)+".npy")
        return 

    def CreateNetwork(self, Hidden_layer, fname, Regularizer, name):

        self.T=int(self.y.shape[0]*9/10)
        self.T=((int)(self.T/32))*32
        Model=keras.Sequential()
        for i in range(len(Hidden_layer)):
            Model.add(keras.layers.Dense(Hidden_layer[i], kernel_initializer='random_uniform',kernel_regularizer=Regularizer, activation=fname))
        Model.add(keras.layers.Dense(self.x.shape[1], activation='linear'))
        self.model[name]=Model
        return

    def TrainNetwork(self, Optimizer, ep, name, los='mse'):
        self.model[name].compile(optimizer=Optimizer, loss=los, metrics=['mae'])
        self.model[name].fit(self.y[:self.T,:], self.x[:self.T,:], epochs=ep, batch_size=16, validation_data=(self.y[self.T:,:], self.x[self.T:,:]))
        return