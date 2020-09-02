#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from Training2D import Trainer2D
from Training2D import RectangleTrainer
import tensorflow as tf
from tensorflow import keras
from Data_Generation import *
import keras.backend as K


# In[ ]:


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
        dd=np.reshape(dd, (m,n))
        dd=np.reshape(dd, m*n)
        depth.append(dd)
        bathytime.append(filename)
    
depth=np.array(depth)


num_x=(int)(n)
num_y=(int)(m)
num_x=75
num_y=51
num_t=3
num_sample=100
ny_s=10
nx_s=10
freq=40

nug=0.4
r=0.07
p=1
param={'psill':p, 'range': r, 'nugget':nug}

river=[]
for i in range(239):
    rr=np.reshape(depth[i,:], (51, 75))
    rr=rr[0:num_y, 0:num_x]
    river.append(rr)

#Tn=RectangleTrainer(num_x, num_y, num_t, num_sample, ny_s, nx_s, freq, param, river)
Tn=Trainer2D(num_x, num_y, num_t, num_sample, ny_s, nx_s, freq, param, river)


# In[ ]:


label, image=Tn.LoadData(109)
R0 = 0.01*np.identity((Tn.y.shape[1]))
R0[48:,48:]*=0.1
dataset=[image, label]

def normalize(data):
    
    MAX=np.max(data, axis=0)
    MIN=np.min(data, axis=0)
    MEAN=np.mean(data,axis=0)
    STD=np.std(data,axis=0)
    
    return np.divide((data-MEAN),(STD))

dataset=[normalize(image), normalize(label)]
IMAGESTD=np.std(image,axis=0)
IMAGEMEAN=np.mean(image,axis=0)
LABELSTD=np.std(label,axis=0)
LABELMEAN=np.mean(label,axis=0)
#dataset[1]+=np.random.normal(0,0.05, size=dataset[1].shape)


# In[ ]:



# define the standalone discriminator model
def define_discriminator(in_shape=(51,75,1), n_classes=10):
    # label input
    in_label = Input(shape=(96,))
    # embedding for categorical input
    #li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1] 
    #in_label = Reshape((13*19))
    li = Dense(n_nodes)(in_label)
    #print(li.shape)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1],1))(li)
    # image input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
    # label input
    in_label = Input(shape=(96,))
    # embedding for categorical input
    #li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    a=7
    b=9
    n_nodes = a * b
    li = Dense(n_nodes)(in_label)
#    li = Dense(n_nodes, activation='relu')(in_label)    
#    li = Dense(48*32, activation='relu')(in_label)   
    # reshape to additional channel
    li = Reshape((a, b, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 128 * a * b
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((a, b, 128))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    	# upsample to 14x14
    gen = Conv2DTranspose(128, (4,4), strides=(3,3), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    	# upsample to 28x28
    gen = Conv2DTranspose(128, (4,4), strides=(3,3), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (a,b), activation='tanh', padding='same')(gen)
    out_layer = Flatten()(out_layer)
    out_layer = Dense(3825)(out_layer)
    out_layer = Reshape((51, 75,1))(out_layer)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], [gan_output, gen_output])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mse'], optimizer=opt, loss_weights=[1, 0.1])
	return model
 
# load fashion mnist images
def load_real_samples():
	# load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X, trainy]
 
# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(dataset, latent_dim, n_samples, n_classes=10):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
	# generate points in the latent space
    x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
    labels_input = labels[ix] 
    images_input = images[ix]
    return [z_input, labels_input, images_input]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(dataset, generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input, images_input = generate_latent_points(dataset, latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(dataset, g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input, images_input] = generate_latent_points(dataset, latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], [y_gan, images_input])
			# summarize loss on this batch
			g_loss
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss[0]+0.1*g_loss[1]))
	# save the generator model
	g_model.save('FixedMagnitudeSizeGenerator.h5')
	d_model.save('FixedMagnitudeSizeDiscriminator.h5')


# In[ ]:


dataset[0]=np.reshape(dataset[0],(dataset[0].shape[0], num_y, num_x, 1))
# size of the latent space
latent_dim = 59
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
#dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100  , n_batch=128)


# In[ ]:


train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100  , n_batch=128)
g_model.save('cgan_generator_nearshore_sparsityx=y=10-Average.h6')

