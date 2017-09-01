# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:32:10 2017

@author: zhaoxm
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist

from gan_model import *
from gan_solver import *
from utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_type = 'BIGAN'
N = 36
D = 28 * 28
hidden_num = 128
z_dim = 64
init_learning_rate = 1e-3
max_iter = 100000
verbose_interval = 1000
show_interval = 1000
snapshot = 1000
K_critic = 1

train_dir = './data/mnist'
train_data = mnist.input_data.read_data_sets(train_dir).train

tf.reset_default_graph()
data = tf.placeholder(tf.float32, [None, D])

if model_type == 'Vanilla':
    model = VanillaGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'Wasserstein':
    use_gp = True
    model = WassersteinGAN(data, hidden_num=hidden_num, z_dim=z_dim, use_gp=use_gp)
    if use_gp:
        train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver]
    else:
        train_op = WassersteinSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver, train_op.grad_clip]
elif model_type == 'EMGAN':
    model = EMGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'EBGAN':
    model = EBGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'BEGAN':
    model = BEGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver, model.balance, model.k_update]
elif model_type == 'BIGAN':
    K_critic = 1
    model = BIGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
else:
    raise NotImplementedError('model_type is wrong.')
    
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))
saver = tf.train.Saver(max_to_keep=20)
init = tf.global_variables_initializer()
sess.run(init)
#saver.restore(sess, "./trial/trial-100000")
for iter in range(max_iter):
    for k in range(K_critic):
        x_batch, _ = train_data.next_batch(N)
        sess.run(d_fetches, feed_dict={data:x_batch})
#    x_batch, _ = train_data.next_batch(N)
    sess.run(train_op.g_solver, feed_dict={data:x_batch})
    
    if iter % verbose_interval == 0:
        d_loss, g_loss, lr = sess.run([model.d_loss, model.g_loss, train_op.learning_rate], feed_dict={data:x_batch})
        print('iter=%d, lr=%f, d_loss=%f, g_loss=%f') % (iter, lr, d_loss, g_loss)
        if model_type == 'BEGAN':
            messure, kt = sess.run([model.messure, model.kt], feed_dict={data:x_batch})
            print('messure=%f, k=%f') % (messure, kt)
        
    if iter % show_interval == 0:
        sample_batch, _ = train_data.next_batch(16)
        samples = sess.run(model.g_data, {data:sample_batch})
        fig = grid_plot(samples)
#        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    if (iter) % snapshot == 0:
        saver.save(sess, './trial/trial', global_step=iter)
