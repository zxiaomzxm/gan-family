# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:40:45 2017

@author: zhaoxm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def xrecons_grid(X,B,A):
	"""
	plots canvas for single time step
	X is x_recons, (batch_size x img_size)
	assumes features = BxA images
	batch is assumed to be a square number
	"""
	padsize=1
	padval=.5
	ph=B+2*padsize
	pw=A+2*padsize
	batch_size=X.shape[0]
	N=int(np.sqrt(batch_size))
	X=X.reshape((N,N,B,A))
	img=np.ones((N*ph,N*pw))*padval
	for i in range(N):
		for j in range(N):
			startr=i*ph+padsize
			endr=startr+B
			startc=j*pw+padsize
			endc=startc+A
			img[startr:endr,startc:endc]=X[i,j,:,:]
	return img

def plot_canvases(C, prefix='test', interactive=True):
	batch_size,img_size=C.shape
	C[C < 0] = 0
	C[C > 1] = 1
	B=A=int(np.sqrt(img_size))
	if interactive:
		f,arr=plt.subplots(1,1)
	for t in range(1):
		img=xrecons_grid(C,B,A)
		if interactive:
			arr.matshow(img, cmap=plt.cm.Blues)
			arr.set_xticks([])
			arr.set_yticks([])
			plt.show()
		else:
			f = plt.figure(figsize=[10,10])
			plt.matshow(img,cmap=plt.cm.gray, fignum=0)
			plt.axis('off')
			imgname='%s_%03d.png' % ('img/'+prefix,t) # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
			plt.savefig(imgname)
			print(imgname)
   
   
def grid_plot(C):
    batch_size,img_size=C.shape
    C[C < 0] = 0
    C[C > 1] = 1
    B=A=int(np.sqrt(batch_size))
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(B, A)
    gs.update(wspace=0.1, hspace=0.1)

    for i, sample in enumerate(C):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Blues')
    return fig