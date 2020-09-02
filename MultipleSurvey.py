#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from Training2D import Trainer2D
import tensorflow as tf
from tensorflow import keras
from Data_Generation import *
import keras.backend as K


# In[2]:


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


# In[3]:


path='SurveyFull/'
depth=[]
bathytime=[]
i=0
for filename in glob.glob(os.path.join(path, '*.txt')):
    i+=1
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
#depth=np.vstack(depth)
# SampleBathy=[]
# SampleTime=[]
# for ii in range(4):
#     for jj in range(depth.shape[0]):
#         #if (bathytime[jj][15:19]=='2000'):
# #        SampleBathy.append(np.reshape(depth[jj],(m,n)))
# #        SampleTime.append(bathytime[jj][15:23]) 
# #        print(bathytime[jj][15:23]) 


# In[4]:


#### Create forward map H
from scipy.sparse import coo_matrix
import cvxpy as cp
####
num_x=75
num_y=51
num_t=3
num_sample=80
ny_s=10
nx_s=10
r=0.075
rivers=[]
for i in range(239):
    rr=np.reshape(depth[i,:num_y*num_x], (num_y, num_x))
    rivers.append(rr)
####
row = []
col = []
data = []
tt = np.zeros((m,n))
tt[::ny_s,::nx_s]=1
M1 = tt[::ny_s,::nx_s].size
count=-1
for i in range(m):
    for j in range(n):
        if (tt[i,j]==1):
            count+=1
            row.append(count)
            col.append(i*n+j)
            data.append(1)
            
ay=(int)((num_y+ny_s-1)/ny_s)
ax=(int)((num_x+nx_s-1)/nx_s)
Temp=np.zeros((ay,ax))
Area=np.zeros((ay,ax))


for i in range(num_y):
    for j in range(num_x):
        ii = (int)(i/ny_s)
        jj = (int)(j/nx_s)
        Area[ii,jj]+=1
for i in range(num_y):
    for j in range(num_x):
        ii = (int)(i/ny_s)
        jj = (int)(j/nx_s)
        row.append(ii*ax+jj+M1)
        col.append(i*n+j)
        data.append(1/Area[ii,jj])
        #Area[ii,jj]+=1
            
#%% L matrix for total variation prior
rowL = []
colL = []
dataL = []
count = -1
for i in range(m):
    for j in range(n):
        if (i>0):
            count+=1
            rowL.append(count)
            colL.append(i*n+j)
            dataL.append(1)
            rowL.append(count)
            colL.append((i-1)*n+j)
            dataL.append(-1)
        if (i<m-1):
            count+=1
            rowL.append(count)
            colL.append(i*n+j)
            dataL.append(1)
            rowL.append(count)
            colL.append((i+1)*n+j)
            dataL.append(-1)
        if (j>0):
            count+=1
            rowL.append(count)
            colL.append(i*n+j)
            dataL.append(1)
            rowL.append(count)
            colL.append((i)*n+j-1)
            dataL.append(-1)
        if (j<n-1):
            count+=1
            rowL.append(count)
            colL.append(i*n+j)
            dataL.append(1)
            rowL.append(count)
            colL.append(i*n+j+1)
            dataL.append(-1)
            
            
rowL=np.array(rowL)
colL=np.array(colL)
dataL=np.array(dataL)
L = coo_matrix((dataL, (rowL, colL)), shape=(2*4+(m-2+n-2)*2*3+((m-2)*(n-2))*4, num_x*num_y))  
L = L.astype(float)
row=np.array(row)
col=np.array(col)
data=np.array(data)
H = coo_matrix((data, (row, col)), shape=(M1+ax*ay, num_x*num_y))
Tn=Trainer2D(num_x, num_y, num_sample, ny_s, nx_s, r, rivers, H)
#### Setting up covariance matrices
from scipy.linalg import null_space
x_coord, y_coord=np.meshgrid(Tn.xx2,Tn.yy2, sparse=False)
r=0.75
covar= lambda x : np.exp(-(x)/(r))
x_coord=np.reshape(x_coord, (num_y*num_x,1))
y_coord=np.reshape(y_coord, (num_y*num_x,1))
Q0=Cov(x_coord, y_coord, covar)
X = np.ones((num_y * num_x, 1)) / np.sqrt(num_y * num_x)
R0 = 0.01*np.identity((M1+ax*ay))
R0[48:,48:]*=0.1
T = null_space((H.dot(X)).T).T
#uq_k = np.reshape(np.diag(Qsy),(51,75))**(0.5)
C1 = T.dot(H.dot(Q0.dot(H.T.dot(T.T))))
C2 = T.dot(R0.dot(T.T))


# In[5]:


#Tn.GenerateData(123)
Tn.LoadData(109)
Tn.y += np.random.multivariate_normal(np.zeros(96), R0, Tn.y.shape[0])


# In[6]:


Hidden_layer=[2000, 2000]
fname='relu'
Regularizer=0.000
Reg=keras.regularizers.l2(Regularizer)
los='mse'


# In[7]:


Tn.CreateNetwork(Hidden_layer, fname, Reg, 'L2')
Tn.TrainNetwork(tf.compat.v1.train.AdamOptimizer(0.001), 8, 'L2',los)
Tn.TrainNetwork(tf.compat.v1.train.AdamOptimizer(0.0001), 2, 'L2',los)
Tn.TrainNetwork(tf.compat.v1.train.AdamOptimizer(0.0000001), 2, 'L2', los)


# In[8]:


from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import matplotlib.pyplot as plt
cgan_model = load_model('FixedMagnitudeSizeGenerator.h5')

def generate_latent_points(dataset, latent_dim, n_samples, n_classes=10):
    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
	# generate points in the latent space
    x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
    labels2 = labels[ix] 
    return [z_input, labels2]
IMAGESTD=np.std(Tn.x,axis=0)
IMAGEMEAN=np.mean(Tn.x,axis=0)
LABELSTD=np.std(Tn.y,axis=0)
LABELMEAN=np.mean(Tn.y,axis=0)


# In[9]:


def plot_compare(d, dd, n, norm, filename, Save=False, Error=True, ind=0, v1=0, v2=7):
    if (norm=='mse'):
        f=lambda x,y: np.mean((x-y)**2)**(0.5)
    else:
        f=lambda x,y: np.mean(np.abs(x-y))
    err=np.zeros(len(d))
    fig = plt.figure(figsize=(10,12))
    iplot=420
    for i in range(len(d)):
        iplot+=1
        err[i]=f(d[i],dd)
        s=n[i]
        if (Error):
            if (n[i] != 'Groundtruth' and n[i] != 'Reference'):
                s+=": "+norm+"= "+np.format_float_scientific(err[i],precision=2, exp_digits=1)
        ax = fig.add_subplot(iplot)
        im = ax.imshow(-d[i], vmin=-v2-2, vmax=-v1)
        ax.set_title(s, fontsize=15)
        ax.set_axis_off()
        cbar = fig.colorbar(im)
    plt.subplots_adjust(hspace=0.15, wspace=0.05)
    if (Save):
        plt.savefig(filename+".pdf", dpi=300, bbox_inches='tight')
    plt.show()
    return


# In[10]:


ind=np.random.randint(0,270)
ind=249
num_sample = 1000
x=(np.copy(np.reshape(depth[ind,:], (m,n))))
x_grid, y_grid=np.meshgrid(Tn.xx2,Tn.yy2,sparse=False)
jump, x0, y0=random_jump_v(x_grid, y_grid)
jump=np.abs(jump)
#jump = np.load('Regjump.npy')
#x+=jump
y = H.dot(np.reshape(x, (num_y * num_x, 1)))
y = y.T + np.random.multivariate_normal(np.zeros(96), R0, 1)
### Compute total variation estimate
y = np.reshape(y,(y.size))
tv = cp.Variable(num_x*num_y)
objective = cp.Minimize(cp.norm(H @ tv - y, 2)**2 + 0.001 * cp.norm(L @ tv, 1))
prob = cp.Problem(objective)
prob.solve()
tv_val = np.array(tv.value)
tv_val = np.reshape(tv_val,(num_y,num_x))
### Compute dnn estimate
y = np.reshape(y,(1,y.size))
#noise = np.random.multivariate_normal(np.zeros(96), R0, 1)
dnn=Tn.model['L2'].predict(np.reshape(y, (1, y.shape[1])))
dnn=np.reshape(dnn,(m,n))
### Compute cGAN estimate
ydim = 59
labeldim = 96
dataset = [Tn.x, Tn.y]
latent_points, labels = generate_latent_points(dataset, ydim, num_sample)
y_nml=(y-LABELMEAN)/LABELSTD
TT=np.zeros((num_sample,labeldim))
TT[0:,:]=y_nml
post_samples  = cgan_model.predict([latent_points, TT])
post_samples  = np.reshape(post_samples, (post_samples.shape[0], num_y * num_x))
post_samples  = IMAGEMEAN+post_samples*IMAGESTD
cgan = np.mean(post_samples,axis=0)
cgan = np.reshape(cgan, (51,75))
### Compute kriging estimate
#####
y = np.reshape(y,(y.size))
z = T.dot(y)
theta1 = np.arange(-0.5, 3.5, 0.04)
theta2 = np.arange(0, 5, 0.05)
loglikelihood = lambda a, b : 0.5 * np.log(np.linalg.det((10**a) * C1 + (10**b) * C2)) + 0.5 * z.dot(np.linalg.solve((10**a) * C1 + (10**b) * C2, z))
result = np.zeros((100,100))
for i, t1 in enumerate(theta1):
    for j, t2 in enumerate(theta2):
        result[i,j] = loglikelihood(t1,t2)
t1 = 10**theta1[np.unravel_index(result.argmin(), result.shape)[0]]
t2 = 10**theta2[np.unravel_index(result.argmin(), result.shape)[1]]
R = t2 * R0
Q = t1 * Q0
Qin = np.linalg.inv(Q)
Rinv = np.linalg.inv(R)
Qsyinv = Qin+H.T.dot((H.T.dot(Rinv.T)).T)
Qsy = np.linalg.inv(Qsyinv)
#####
y = np.reshape(y,(y.size))
mu_gt = np.mean(depth[:239,:], axis=0)
mu = np.reshape(dnn, (num_y*num_x))
uq_k = np.reshape(np.diag(Qsy),(51,75))**(0.5)
kriging = mu_gt + Qsy.dot(H.T.dot(Rinv.dot(y-H.dot(mu_gt))))
kriging = np.reshape(kriging,(num_y,num_x)) 
### Compute dnn-kriging estimate
dnn_kriging = mu + Qsy.dot(H.T.dot(Rinv.dot(y-H.dot(mu))))
dnn_kriging = np.reshape(dnn_kriging,(num_y,num_x))
predictions=[-dnn_kriging,-dnn,-cgan,-kriging, -tv_val, -x]
names=['DNN-Kriging', 'DNN', 'cGAN', 'Kriging', 'TV', 'Reference']
plot_compare(predictions, -x, names, 'mse', 'benchmarks', True, True, 2, -8, 10)


# In[11]:


sui = np.random.normal(0, 1, (num_sample, num_y * num_x))
sui = sui.dot(Tn.L.T)
Hsui = (H.dot(sui.T)).T
Hsui += np.random.multivariate_normal(np.zeros(96), R0, 1000)
yHsui = y - Hsui
########
s_sui = Tn.model['L2'].predict(yHsui) + sui
uq_dnn = np.reshape(np.std(s_sui, axis=0), (51, 75))
plt.imshow(uq_dnn)
plt.axis('off')
plt.title('Uncertainty of DNN', fontsize=15)
plt.colorbar()
plt.figure()
########
uq_cgan = np.reshape(np.std(post_samples,axis=0), (51, 75))
plt.figure()
plt.imshow(uq_cgan, vmax=0.15)
plt.axis('off')
plt.title('Uncertainty of cGAN', fontsize=15)
plt.colorbar()


# In[12]:


# y = np.reshape(y,(y.size))
# z = T.dot(y)
theta1 = np.arange(-0.5, 3.5, 0.04)
theta2 = np.arange(0, 5, 0.05)
loglikelihood = lambda a,b : 0.5 * np.log(np.linalg.det((10**a) * C1 + (10**b) * C2)) + 0.5 * z.dot(np.linalg.solve((10**a) * C1 + (10**b) * C2, z))
result = np.zeros((100,100))
for i, t1 in enumerate(theta1):
    for j, t2 in enumerate(theta2):
        result[i,j] = loglikelihood(t1,t2)
xx, yy = np.meshgrid(theta1, theta2, sparse=False)
plt.figure()
plt.contourf(xx,yy,np.log(result), levels=50)
np.unravel_index(result.argmin(), result.shape)
print(np.unravel_index(result.argmin(), result.shape))


# In[13]:


loc_y =np.argmin(np.abs(Tn.yy2-y0))
loc_y =10
plt.figure(figsize=(8,4))
plt.plot(np.squeeze(Tn.xx2), kriging[loc_y,:], label="Kriging", color='lightcoral')
plt.plot(np.squeeze(Tn.xx2), x[loc_y,:], label="Reference", color='orange')
plt.fill_between(np.squeeze(Tn.xx2), (dnn_kriging+2*uq_k)[loc_y,:], (dnn_kriging-2*uq_k)[loc_y,:], color = 'green', alpha=.5, label='DNN-Kriging')
plt.fill_between(np.squeeze(Tn.xx2), (cgan+2*uq_cgan)[loc_y,:], (cgan-2*uq_cgan)[loc_y,:], color = 'darkslategrey', alpha=.5, label='cGAN')
plt.fill_between(np.squeeze(Tn.xx2), (dnn+2*uq_dnn)[loc_y,:], (dnn-2*uq_dnn)[loc_y,:], color = 'lightblue', alpha=.5, label='DNN')
plt.plot(np.squeeze(Tn.xx2), tv_val[loc_y,:], label='TV', color='yellowgreen')
plt.scatter(np.squeeze(Tn.xx2)[::nx_s], x[loc_y,::nx_s])
plt.legend()
#plt.ylim((-14,16))
plt.title("Across-shore Section",fontsize=20)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig("DirectPointwise+Averageacross-shore16.pdf", bbox_inches='tight')
plt.show()
plt.close()
loc_x = np.argmin(np.abs(Tn.xx2-x0))
loc_x = 30
plt.figure(figsize=(8,4))
plt.plot(np.squeeze(Tn.yy2), kriging[:,loc_x], label="Kriging", color='lightcoral')
plt.plot(np.squeeze(Tn.yy2), x[:,loc_x], label="Reference", color='orange')
plt.fill_between(np.squeeze(Tn.yy2), (dnn_kriging+2*uq_k)[:,loc_x], (dnn_kriging-2*uq_k)[:,loc_x], color = 'green', alpha=.5, label='DNN-Kriging')
plt.fill_between(np.squeeze(Tn.yy2), (cgan+2*uq_cgan)[:,loc_x], (cgan-2*uq_cgan)[:,loc_x], color = 'darkslategrey', alpha=.5, label='cGAN')
plt.fill_between(np.squeeze(Tn.yy2), (dnn+2*uq_dnn)[:,loc_x], (dnn-2*uq_dnn)[:,loc_x], color = 'lightblue', alpha=.5, label='DNN')
plt.plot(np.squeeze(Tn.yy2), tv_val[:,loc_x],label='TV', color='yellowgreen')
plt.scatter(np.squeeze(Tn.yy2)[::ny_s], x[::ny_s,loc_x])
plt.legend()
plt.ylim((-12,6))
yint = np.arange(-10, 5, 2)
plt.title("Along-shore Section", fontsize=20)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig("DirectPointwise+Averagealong-shore16.pdf", bbox_inches='tight')
plt.show()


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (zoomed_inset_axes, inset_axes, InsetPosition,
                                                  mark_inset)


#plt.figure(figsize=(8,4))
fig, ax1 = plt.subplots(figsize=(8,4))
loc_y =np.argmin(np.abs(Tn.yy2-y0))
loc_y =40
#axe1.figure(figsize=(8,4))
ax1.plot(np.squeeze(Tn.xx2), kriging[loc_y,:], label="Kriging", color='lightcoral')
ax1.fill_between(np.squeeze(Tn.xx2), (dnn_kriging+2*uq_k)[loc_y,:], (dnn_kriging-2*uq_k)[loc_y,:], color = 'green', alpha=.5, label='DNN-Kriging')
ax1.fill_between(np.squeeze(Tn.xx2), (cgan+2*uq_cgan)[loc_y,:], (cgan-2*uq_cgan)[loc_y,:], color = 'darkslategrey', alpha=.5, label='cGAN')
ax1.plot(np.squeeze(Tn.xx2), x[loc_y,:], label="Reference", color='orange')
    #if ((i==0) or (i==1) or (i==4)):
ax1.fill_between(np.squeeze(Tn.xx2), (dnn+2*uq_dnn)[loc_y,:], (dnn-2*uq_dnn)[loc_y,:], color = 'lightblue', alpha=.5, label='DNN')
ax1.plot(np.squeeze(Tn.xx2), tv_val[loc_y,:],label='TV', color='yellowgreen')
#plt.plot(np.squeeze(Tn.xx2), dlr[np.argmin(np.abs(Tn.yy2-y0)),:],label="nnrk")
ax1.scatter(np.squeeze(Tn.xx2)[::nx_s], x[loc_y,::nx_s])
ax1.legend()
plt.title("Across-shore Section",fontsize=20)

ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax2 = plt.axes([0.45,0.5,0.25,0.35])
ax2.set_xlim(0.12,0.275)
ax2.set_ylim(-5,-1)
# Manually set the position and relative size of the inset axes within ax1
#ip = InsetPosition(ax1, [0.4,0.6,0.3,0.3])
#ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
# The data: only display for low temperature in the inset figure.

ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax2.plot(np.squeeze(Tn.xx2), kriging[loc_y,:], label="Kriging", color='lightcoral')
ax2.plot(np.squeeze(Tn.xx2), x[loc_y,:], label="Reference", color='orange')
ax2.plot(np.squeeze(Tn.xx2), dnn[loc_y,:],label='DNN mean', color='red')
ax2.scatter(np.squeeze(Tn.xx2)[::nx_s], x[loc_y,::nx_s])

ax2.legend(loc=0, prop={'size': 8})
#plt.ylim((-14,12))

plt.savefig("DirectPointwise+Averageacross-shore16.pdf", bbox_inches='tight')


plt.show()
plt.close()


# In[ ]:


rdl = np.zeros(15)
rk = np.zeros(15)
rgan = np.zeros(15)

for k,ind in enumerate(np.arange(240,255)):
    x=(np.copy(np.reshape(depth[ind,:], (m,n))))
    y = H.dot(np.reshape(x, (num_y * num_x, 1)))
    ### Compute dnn estimate
    y = np.reshape(y,(1,y.size))
    noise = np.random.multivariate_normal(np.zeros(96), R0, 1)
    dnn=Tn.model['L1'].predict(np.reshape(y + noise, (1, y.shape[1])))
    dnn=np.reshape(dnn,(m,n))
    ### Compute cGAN estimate
    dataset = [Tn.x, Tn.y]
    latent_points, labels = generate_latent_points(dataset, ydim, num_sample)
    y_nml=(y-LABELMEAN)/LABELSTD
    TT=np.zeros((num_sample,labeldim))
    TT[0:,:]=y_nml
    post_samples  = cgan_model.predict([latent_points, TT])
    post_samples  = np.reshape(post_samples, (post_samples.shape[0], num_y * num_x))
    post_samples  = IMAGEMEAN+post_samples*IMAGESTD
    cgan = np.mean(post_samples,axis=0)
    cgan = np.reshape(cgan, (51,75))
    ### Compute kriging estimate
    #####
    y = np.reshape(y,(y.size))
    z = T.dot(y)
    def f(theta):
        return 0.5 * np.log(np.linalg.det(theta[0] * C1 + theta[1] * C2)) + 0.5 * z.dot(np.linalg.solve(theta[0] * C1 + theta[1] * C2, z))
    theta1 = np.arange(-0.5, 3.5, 0.04)
    theta2 = np.arange(0, 5, 0.05)
    loglikelihood = lambda x,y : 0.5 * np.log(np.linalg.det((10**x) * C1 + (10**y) * C2)) + 0.5 * z.dot(np.linalg.solve((10**x) * C1 + (10**y) * C2, z))
    result = np.zeros((100,100))
    for i, t1 in enumerate(theta1):
        for j, t2 in enumerate(theta2):
            result[i,j] = loglikelihood(t1,t2)
    t1 = 10**theta1[np.unravel_index(result.argmin(), result.shape)[0]]
    t2 = 10**theta2[np.unravel_index(result.argmin(), result.shape)[1]]
    R = t2 * R0
    Q = t1 * Q0
    Qin = np.linalg.inv(Q)
    Rinv = np.linalg.inv(R)
    Qsyinv = Qin+H.T.dot((H.T.dot(Rinv.T)).T)
    Qsy = np.linalg.inv(Qsyinv)
    #####
    mu_gt = np.mean(depth[:240,:], axis=0)
    y = np.reshape(y,(y.size))
    mu = np.reshape(dnn, (num_y*num_x))
    kriging = mu_gt + Qsy.dot(H.T.dot(Rinv.dot(y-H.dot(mu_gt))))
    kriging = np.reshape(kriging,(num_y,num_x)) 
    rdl[k]=np.mean((x-dnn)**2)**(0.5)
    rk[k]=np.mean((x-kriging)**2)**(0.5)
    rgan[k]=np.mean((x-cgan)**2)**(0.5)


# In[ ]:


xxx = np.arange(15)
width = 0.3
plt.figure(figsize=(8,6))
plt.bar(xxx-width*5/3, rdl, width, label='DNN')
plt.bar(xxx-width*2/3, rgan, width, label='cGAN')
plt.bar(xxx+width/3, rk, width, label='Kriging')
plt.xlabel('FRF Surveys', fontsize=15)
plt.ylabel('Root Mean Squared Error', fontsize=15)
plt.legend()
plt.savefig('OverallBenchmark.pdf', bbox_inches='tight')
print(np.mean(rdl))
print(np.mean(rgan))
print(np.mean(rk))


# In[ ]:


fig = plt.figure(figsize=(10,12))
iplot=420

ax = fig.add_subplot(421)
im = ax.imshow(uq_cgan, vmax=0.15)
ax.set_title('Uncertainty of cGAN', fontsize=15)
ax.set_axis_off()
cbar = fig.colorbar(im)
ax = fig.add_subplot(422)
im = ax.imshow(uq_k)
ax.set_title('Uncertainty of Kriging', fontsize=15)
ax.set_axis_off()
cbar = fig.colorbar(im)

ax = fig.add_subplot(423)
im = ax.imshow(uq_dnn)
ax.set_title('Uncertainty of DNN', fontsize=15)
ax.set_axis_off()
cbar = fig.colorbar(im)
#cb_ax = fig.add_axes([0.95, 0.02, 0.02, 0.85])
#cbar = fig.colorbar()
plt.subplots_adjust(hspace=0.15, wspace=0.05)
plt.savefig('Uncertainty.pdf', bbox_inches='tight')
plt.show()

