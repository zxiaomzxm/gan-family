# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:24:49 2017

@author: zhaoxm
"""

import tensorflow as tf
from skimage import io
from gan_model import *
from gan_solver import *
from utils import *
import seaborn as sb
from sampler import generate_lut, sample_2d
from visualizer import GANDemoVisualizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_type = 'Vanilla'
N = 20 00
D = 2
hidden_num = 128
z_dim = 2
init_learning_rate = 1e-3
max_iter = 100000
verbose_interval = 100
show_interval = 1000
snapshot = 1000


train_dir = './img/triangle.jpg'
density_img = io.imread(train_dir, True)
lut_2d = generate_lut(density_img)
visualizer = GANDemoVisualizer('GAN 2D Example Visualization')

tf.reset_default_graph()
data = tf.placeholder(tf.float32, [None, D])

if model_type == 'Vanilla':
    K = 1
    model = VanillaGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'Wasserstein':
    K = 1
    use_gp = True
    model = WassersteinGAN(data, hidden_num=hidden_num, z_dim=z_dim, use_gp=use_gp)
    if use_gp:
        train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver]
    else:
        train_op = WassersteinSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver, train_op.clip_grad]
elif model_type == 'EMGAN':
    K = 1
    use_gp = True
    model = EMGAN(data, hidden_num=hidden_num, z_dim=z_dim, use_gp=use_gp, embedding_dim=10)
    if use_gp:
        train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver]
    else:
        train_op = WassersteinSolver(model, init_learning_rate=init_learning_rate)
        d_fetches = [train_op.d_solver, train_op.clip_grad]
elif model_type == 'EBGAN':
    K = 1
    model = EBGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver]
elif model_type == 'BEGAN':
    K = 1
    model = BEGAN(data, hidden_num=hidden_num, z_dim=z_dim)
    train_op = BaseSolver(model, init_learning_rate=init_learning_rate)
    d_fetches = [train_op.d_solver, model.balance, model.k_update]
else:
    raise NotImplementedError('model_type is wrong.')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=20)
init = tf.global_variables_initializer()
sess.run(init)
#saver.restore(sess, "./trial/trial-100000")
for iter in range(max_iter):
    for k in range(K):
        x_batch = sample_2d(lut_2d, N) - 0.5
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
        real_samples = sample_2d(lut_2d, 2000)
        gen_samples = sess.run(model.g_data, feed_dict={data:real_samples}) + 0.5
        visualizer.draw(real_samples, gen_samples)
        plt.show()
        print ''
    
    if (iter) % snapshot == 0:
        saver.save(sess, './trial/trial', global_step=iter)
