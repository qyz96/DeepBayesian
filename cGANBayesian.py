# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:30:15 2019

@author: yizhou
"""



# example of training an conditional gan on the fashion mnist dataset
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
 
#%%
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


num_x=(int)(n)
num_y=(int)(m)
num_x=64
num_y=48
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
#%%


#label, image=Tn.GenerateData(106)
label, image=Tn.LoadData(106,0)

###106   107+TakeAverage
###107   Similar to 108, Sparsity increased from 4 to 10 in each direction,
##JUMP and Variation, random size, magnitude and location
###108   JUMP only and Variation Only, no constant, 2-3 Samples : Good Results! Saved Now as cgan_generator_nearshore.h5
#%%
dataset=[image, label]


def normalize(data):
    
    MAX=np.max(data, axis=0)
    MIN=np.min(data, axis=0)
    MEAN=np.mean(data,axis=0)
    
    return np.divide((data-MEAN),((MAX-MIN)/2))


dataset=[normalize(image), normalize(label)]


#dataset[0]+=np.random.normal(0,0.04, size=dataset[0].shape)
dataset[1]+=np.random.normal(0,5, size=dataset[1].shape)


#%%

IMAGEMAX=np.max(image, axis=0)
IMAGEMIN=np.min(image, axis=0)
IMAGEMEAN=np.mean(image,axis=0)
LABELMAX=np.max(label, axis=0)
LABELMIN=np.min(label, axis=0)
LABELMEAN=np.mean(label,axis=0)
#%%



# define the standalone discriminator model
def define_discriminator(in_shape=(48,64,1), n_classes=10):
    # label input
    in_label = Input(shape=(24,))
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
    in_label = Input(shape=(24,))
    # embedding for categorical input
    #li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    a=12
    b=16
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
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    	# upsample to 28x28
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (a,b), activation='tanh', padding='same')(gen)
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
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
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
    labels2 = labels[ix] 
    return [z_input, labels2]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(dataset, generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(dataset, latent_dim, n_samples)
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
			[z_input, labels_input] = generate_latent_points(dataset, latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# save the generator model
	g_model.save('cgan_generator_nearshore_sparsityx=y=10-Average.h5')
#%%
# size of the latent space
latent_dim = 100
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

#%%


# example of loading the generator model and generating images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import matplotlib.pyplot as plt


#%%

nn=np.random.randint(0, 10000)


plt.imshow(np.reshape(dataset[0][nn,:], (48,64)))
#plt.imshow(np.reshape(depth[240,:], (51,75)))
plt.colorbar()
#%%
# load model
model = load_model('cgan_generator_nearshore_sparsityx=y=10-Average.h5')
# generate images

# specify labels
#
nn=244
labeldim=59
#nn=np.random.randint(0, 10000)
#Label=dataset[1][nn,:]
#Label=np.reshape(Label,(1,192))
x,y=np.meshgrid(Tn.xx2,Tn.yy2,sparse=False)
jump,x0, y0=random_jump_v(x,y,1,0.15, 0.3, 24)
jump=np.abs(jump)


NumofSampling=100
latent_points, labels = generate_latent_points(dataset, 100, NumofSampling)
GT=np.copy(np.reshape(depth[nn,:], (51,75)))
GT=GT[0:num_y, 0:num_x]
x,y=np.meshgrid(Tn.xx2,Tn.yy2,sparse=False)

GT+=jump[0:num_y, 0:num_x]
Label=np.reshape(TakeAverage(GT, Tn.num_y, Tn.num_x, Tn.ny_s, Tn.nx_s), (1,24))
Label=Concat(Tn.Sparse(GT), Label)
Label=(Label-LABELMEAN)/((LABELMAX-LABELMIN)/2)
TT=np.zeros((NumofSampling,labeldim))
TT[0:,:]=Label
#plt.imshow(np.reshape(Label, ()))
# generate images
X  = model.predict([latent_points, TT])

X=IMAGEMEAN+X*((IMAGEMAX-IMAGEMIN)/2)
#GT=np.reshape(dataset[0][nn,:], (48,64,1))
#GT=IMAGEMEAN+GT*((IMAGEMAX-IMAGEMIN)/2)
##GT[::ny_s,::nx_s]=10
#GT=np.reshape(GT,(48,64))
print((np.mean(np.abs(np.reshape(np.mean(X, axis=0), (48,64))-GT)**1))**(1))

r=0.4
p=0.2
s=1
covar2= lambda x : p+s*x
x_coord, y_coord=np.meshgrid(Tn.xx2,Tn.yy2, sparse=False)
data_out=kriging(x_coord, y_coord, Tn.Sparse(GT), Tn.ny_s, Tn.nx_s, covar2)
data_out=np.reshape(data_out,(Tn.num_y, Tn.num_x))

print((np.mean(np.abs(data_out-GT)))**(1))



A=-10
B=10
fig, axeslist = plt.subplots(ncols=2, nrows=2, figsize=(9,9))
im=axeslist.ravel()[0].imshow(np.reshape(np.mean(X, axis=0), (48,64)), vmin=A, vmax=B)
axeslist.ravel()[0].set_title('Mean of cGAN')
axeslist.ravel()[0].set_axis_off()
im=axeslist.ravel()[1].imshow(GT,vmin=A, vmax=B)
axeslist.ravel()[1].set_title('Groundtruth')
axeslist.ravel()[1].set_axis_off()
im=axeslist.ravel()[2].imshow(data_out, vmin=A, vmax=B)
axeslist.ravel()[2].set_title('OK')
axeslist.ravel()[2].set_axis_off()
cb_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cb_ax)
im=axeslist.ravel()[3].imshow(np.reshape(np.var(X, axis=0), (48,64)), vmin=0, vmax=0.1)
axeslist.ravel()[3].set_title('Variance of cGAN')
axeslist.ravel()[3].set_axis_off()

plt.subplots_adjust(hspace=0.1, wspace=0)
#plt.savefig('cGAN-Sparsity=10TESTCASEnn=244-2.pdf')


#%%

plt.imshow(np.reshape(np.var(X, axis=0), (48,64)), vmin=0, vmax=0.1)
plt.title('Variance of cGAN')
plt.colorbar()

#%%



plt.plot(np.squeeze(Tn.xx2), data_out[np.argmin(np.abs(Tn.yy2-y0)),:], label="OK")
#plt.plot(np.squeeze(Tn.xx2), dl[np.argmin(np.abs(Tn.yy2-y0)),:], label="L2")
#plt.plot(np.squeeze(Tn.xx2), dd[np.argmin(np.abs(Tn.yy2-y0)),:], label="groundtruth")

plt.plot(np.squeeze(Tn.xx2), np.reshape(np.mean(X, axis=0), (48,64))[np.argmin(np.abs(Tn.yy2-y0)),:],label='cGAN')
plt.plot(np.squeeze(Tn.xx2), GT[np.argmin(np.abs(Tn.yy2-y0)),:],label='Groundtruth')
#plt.plot(np.squeeze(Tn.xx2), dlr[np.argmin(np.abs(Tn.yy2-y0)),:],label="nnrk")
plt.scatter(np.squeeze(Tn.xx2)[::nx_s], GT[np.argmin(np.abs(Tn.yy2-y0)),::nx_s])
plt.legend()
plt.title("across-shore section")
#plt.savefig("Graphs/cGAN-Sparsity=10TESTCASEnn=244-2cross-shore.pdf")
plt.show()
plt.close()


#%%
plt.plot(np.squeeze(Tn.yy2), data_out[:,np.argmin(np.abs(Tn.xx2-x0))], label="OK")
#plt.plot(np.squeeze(Tn.yy2), dl[n:,np.argmin(np.abs(Tn.xx2-x0))], label="L2")
#plt.plot(np.squeeze(Tn.yy2), dd[:,np.argmin(np.abs(Tn.xx2-x0))], label="groundtruth")

plt.plot(np.squeeze(Tn.yy2), np.reshape(np.mean(X, axis=0), (48,64))[:,np.argmin(np.abs(Tn.xx2-x0))],label='cGAN')
plt.plot(np.squeeze(Tn.yy2), GT[:,np.argmin(np.abs(Tn.xx2-x0))],label='groundtruth')
plt.scatter(np.squeeze(Tn.yy2)[::ny_s], GT[::ny_s,np.argmin(np.abs(Tn.xx2-x0))])
plt.legend()
plt.title("along-shore section")
#plt.savefig("Graphs/cGAN-Sparsity=10TESTCASEnn=244-2along-shore.pdf")
plt.show()



