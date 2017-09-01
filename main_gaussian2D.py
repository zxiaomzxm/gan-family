# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:06:05 2017

@author: zhaoxm
"""

import tensorflow as tf
from skimage import io
from gan_model import *
from gan_solver import *
from utils import *
import seaborn as sb
from mix_gaussian2D import create_mixgaussian2D
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_type = 'fGAN'
N = 500
D = 2
hidden_num = 8
z_dim = 2
init_learning_rate = 1e-3
max_iter = 100000
verbose_interval = 100
show_interval = 1000
snapshot = 1000

tf.reset_default_graph()

x_data = create_mixgaussian2D(num_components=8)
x_data_sample = x_data.sample(N)
test_sample = x_data.sample(1000)

data = tf.placeholder(tf.float32, [None, D])

if model_type == 'Vanilla':
    K_critic = 1
    model = VanillaGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'Wasserstein':
    K_critic = 1
    use_gp = True
    model = WassersteinGAN(data, hidden_num=hidden_num, z_dim=z_dim, use_gp=use_gp)
    if use_gp:
        train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver]
    else:
        train_op = WassersteinSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver, train_op.clip_grad]
elif model_type == 'EMGAN':
    K_critic = 1
    use_gp = True
    model = EMGAN(data, hidden_num=hidden_num, z_dim=z_dim, use_gp=use_gp, embedding_dim=1)
    if use_gp:
        train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver]
    else:
        train_op = WassersteinSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver, train_op.clip_grad]
elif model_type == 'LSGAN':
    K_critic = 1
    model = LSGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver] 
elif model_type == 'EBGAN':
    K_critic = 1
    model = EBGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'BEGAN':
    K_critic = 1
    model = BEGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver, model.balance, model.k_update]
elif model_type == 'BIGAN':
    K_critic = 1
    model = BIGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'fGAN':
    K_critic = 1
    model = fGAN(data, hidden_num=hidden_num, z_dim=z_dim, f_divergence='TV')
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver] 
else:
    raise NotImplementedError('model_type is wrong.')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=20)
init = tf.global_variables_initializer()
sess.run(init)
#saver.restore(sess, "./trial/trial-34000")
for iter in range(max_iter):
    for k in range(K_critic):
        x_batch = sess.run(x_data_sample)
        sess.run(d_fetches, feed_dict={data:x_batch})
    sess.run(train_op.g_solver, feed_dict={data:x_batch})
    
    if iter % verbose_interval == 0:
        d_loss, g_loss, lr = sess.run([model.d_loss, model.g_loss, train_op.learning_rate], feed_dict={data:x_batch})
        print('iter=%d, lr=%f, d_loss=%f, g_loss=%f') % (iter, lr, d_loss, g_loss)
        if model_type == 'BEGAN':
            messure, kt = sess.run([model.messure, model.kt], feed_dict={data:x_batch})
            print('messure=%f, k=%f') % (messure, kt)
        
    if iter % show_interval == 0:
        real_samples = sess.run(test_sample)
        gen_samples = sess.run(model.g_data, feed_dict={data:real_samples})
        if model_type == 'BIGAN':
            gen_z, recons_samples = sess.run([model.z_enc, model.data_recons], feed_dict={data:real_samples})
        if model_type == 'EBGAN' or model_type == 'BEGAN':
            disc_samples = sess.run(model.d_neg_recons, feed_dict={data:real_samples})
        f, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(13.5, 4))
        plt.axis('equal')
        ax1.plot(real_samples[:,0], real_samples[:,1], 'b.')
        ax1.hold('True')
        ax1.plot(gen_samples[:,0], gen_samples[:,1], 'g.')
        if model_type == 'EBGAN' or model_type == 'BEGAN':
            ax1.plot(disc_samples[:,0], disc_samples[:,1], 'r.')
            sb.kdeplot(disc_samples, ax=ax2, cmap='Blues', n_levels=100, shade=True, clip=[[-6, 6]]*2)
        elif model_type == 'BIGAN':
            ax1.plot(recons_samples[:,0], recons_samples[:,1], 'r.')
            sb.kdeplot(gen_z, ax=ax2, cmap='Blues', n_levels=100, shade=True, clip=[[-6, 6]]*2)
        else:
            sb.kdeplot(real_samples, ax=ax2, cmap='Blues', n_levels=100, shade=True, clip=[[-6, 6]]*2)
        sb.kdeplot(gen_samples, ax=ax3, cmap='Blues', n_levels=100, shade=True, clip=[[-6, 6]]*2)
       
        ax1.hold('False')
        ax1.axis([-1.5, 1.5, -1.5, 1.5])
        ax2.axis([-1.5, 1.5, -1.5, 1.5])
        ax3.axis([-1.5, 1.5, -1.5, 1.5])
        
        plt.show()
        print ''
    
    if (iter) % snapshot == 0:
        saver.save(sess, './trial/trial', global_step=iter)
