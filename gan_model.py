# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:49:52 2017

@author: zhaoxm
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

class _BaseGAN(object):
    """Abstract object representing an GAN model.
    
    """
    def latent_sample(self, prior='norm'):
        if prior == 'uniform':
            z_prior = tf.random_uniform([self.N, self.z_dim], -1, 1)
        elif prior == 'norm':
            z_prior = tf.random_normal([self.N, self.z_dim])
        else:
            NotImplementedError('Wrong prior type')
        return z_prior
        
    def generator(self, z):
        raise NotImplementedError('Abstract Method')
        
    def discriminator(self, x, reuse=None):
        raise NotImplementedError('Abstract Method')
        
    def define_net(self):
        raise NotImplementedError('Abstract Method')
        
    def define_loss(self):
        raise NotImplementedError('Abstract Method')
        
        
class VanillaGAN(_BaseGAN):
    def __init__(self, data, hidden_num, z_dim):
        self.data = data
        self.N = tf.shape(data)[0]
#        self.N = data.get_shape().as_list()[0]
        self.D = data.get_shape().as_list()[1]
        self.hidden_num = hidden_num
        self.z_dim = z_dim
        self.define_net()
        self.define_loss()       
        
    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=z, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            g_data = slim.fully_connected(inputs=p, num_outputs=self.D, activation_fn=tf.identity)
            g_var = tf.contrib.framework.get_variables(vs)
        return g_data, g_var
         
    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=x, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            d_prob = slim.fully_connected(inputs=p, num_outputs=1, activation_fn=tf.nn.sigmoid)
            d_var = tf.contrib.framework.get_variables(vs)
        return d_prob, d_var
        
    def define_net(self):
        z = self.latent_sample()
        self.g_data, self.g_var = self.generator(z)
        self.d_neg_prob, self.d_var = self.discriminator(self.g_data)
        self.d_pos_prob, _ = self.discriminator(self.data, reuse=True)
        
    def define_loss(self):
        self.d_loss = -tf.reduce_mean(tf.log(self.d_pos_prob) + tf.log(1. - self.d_neg_prob))
        self.g_loss = -tf.reduce_mean(tf.log(self.d_neg_prob))
        
class WassersteinGAN(VanillaGAN):
    def __init__(self, data, hidden_num, z_dim, use_gp=False):
        self.use_gp = use_gp
        super(WassersteinGAN, self).__init__(data, hidden_num, z_dim)

    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=x, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            d_prob = slim.fully_connected(inputs=p, num_outputs=1, activation_fn=tf.identity)
            d_var = tf.contrib.framework.get_variables(vs)
        return d_prob, d_var
        
    def define_loss(self):        
        self.d_loss = -tf.reduce_mean(self.d_pos_prob) + tf.reduce_mean(self.d_neg_prob)
        self.g_loss = -tf.reduce_mean(self.d_neg_prob)

        # gradient penalty        
        if self.use_gp:
            lam = 10.0
            eps = tf.random_uniform([self.N, 1], minval=0., maxval=1.)
            X_inter = eps * self.data + (1. - eps) * self.g_data
            grad = tf.gradients(self.discriminator(X_inter, reuse=True)[0], [X_inter])[0]
            grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
            grad_pen = lam * tf.reduce_mean(grad_norm - 1.)**2
            self.d_loss += grad_pen

        
class EMGAN(WassersteinGAN):
    """Embedding GAN
    
    """
    def __init__(self, data, hidden_num, z_dim, use_gp=False, embedding_dim=10):
        self.embedding_dim = embedding_dim
        super(EMGAN, self).__init__(data, hidden_num, z_dim, use_gp)
        
    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=x, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            d_embedding = slim.fully_connected(inputs=p, num_outputs=self.embedding_dim, activation_fn=tf.identity)
            d_embedding = tf.reduce_sum(d_embedding, 1)
            d_var = tf.contrib.framework.get_variables(vs)
        return d_embedding, d_var

class LSGAN(VanillaGAN):
    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=x, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            d_prob = slim.fully_connected(inputs=p, num_outputs=1, activation_fn=tf.identity)
            d_var = tf.contrib.framework.get_variables(vs)
        return d_prob, d_var
        
    def define_loss(self):        
        self.d_loss = 0.5 * (tf.reduce_mean((self.d_pos_prob - 1)**2) + tf.reduce_mean(self.d_neg_prob**2))
        self.g_loss = 0.5 * tf.reduce_mean((self.d_neg_prob - 1)**2)
        
        
class EBGAN(VanillaGAN):    
    def __init__(self, data, hidden_num, z_dim, margin=5.0):
        self.margin = margin
        super(EBGAN, self).__init__(data, hidden_num, z_dim)
        
    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=x, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            d_recons = slim.fully_connected(inputs=p, num_outputs=self.D, activation_fn=tf.identity)
            d_var = tf.contrib.framework.get_variables(vs)
        return d_recons, d_var
        
    def define_net(self):
        z = self.latent_sample()
        self.g_data, self.g_var = self.generator(z)
        self.d_neg_recons, self.d_var = self.discriminator(self.g_data)
        self.d_pos_recons, _ = self.discriminator(self.data, reuse=True)
        
    def mse_calc(self, x, y):
        return tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=1))
        
    def define_loss(self):
        self.d_pos_prob = self.mse_calc(self.data, self.d_pos_recons)
        self.d_neg_prob = self.mse_calc(self.g_data, self.d_neg_recons)
        self.d_loss = self.d_pos_prob + tf.maximum(0., self.margin - self.d_neg_prob)
        self.g_loss = self.d_neg_prob

        
class BEGAN(EBGAN):       
    def __init__(self, data, hidden_num, z_dim, lambda_k=1e-3, gamma=0.5):
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.kt = tf.Variable(0.0, trainable=False, name='k_t')
        super(BEGAN, self).__init__(data, hidden_num, z_dim)
        
    def define_loss(self):
        self.d_pos_prob = self.mse_calc(self.data, self.d_pos_recons)
        self.d_neg_prob = self.mse_calc(self.g_data, self.d_neg_recons)
        self.d_loss = self.d_pos_prob - self.kt * self.d_neg_prob
        self.g_loss = self.d_neg_prob
        
        self.balance = self.gamma * self.d_pos_prob - self.d_neg_prob
        self.messure = self.d_pos_prob + tf.abs(self.balance)
        
        self.k_update = tf.assign(self.kt, tf.clip_by_value(self.kt + self.lambda_k * self.balance, 0, 1))


class BIGAN(VanillaGAN):
    def sample(self, mu, sigma):
        noise = tf.random_normal([self.N, self.z_dim], mean=0.0, stddev=1.0)
        return noise * sigma + mu
            
    def encoder(self, x, reuse=None):
        with tf.variable_scope('encoder', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=x, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            z_enc = slim.fully_connected(inputs=p, num_outputs=self.z_dim, activation_fn=tf.identity)
            e_var = tf.contrib.framework.get_variables(vs)
        return z_enc, e_var
        
#    def generator(self, z):
#        with tf.variable_scope('generator') as vs:
#            p = slim.fully_connected(inputs=z, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
#            g_data = slim.fully_connected(inputs=p, num_outputs=self.D, activation_fn=tf.identity)
#            g_var = tf.contrib.framework.get_variables(vs)
#        return g_data, g_var
#         
    def discriminator_z(self, x, reuse=None):
        with tf.variable_scope('discriminator_z', reuse=reuse) as vs:
            p = slim.fully_connected(inputs=x, num_outputs=self.hidden_num, activation_fn=tf.nn.relu)
            d_prob = slim.fully_connected(inputs=p, num_outputs=1, activation_fn=tf.nn.sigmoid)
            d_var = tf.contrib.framework.get_variables(vs)
        return d_prob, d_var
        
    def define_net(self):
        self.z = self.latent_sample()
        self.g_data, self.g_var = self.generator(self.z)
        self.z_recons, self.e_var = self.encoder(self.g_data)
        self.z_enc, _ = self.encoder(self.data, reuse=True)
        self.data_recons, _ = self.generator(self.z_enc, reuse=True)
        self.g_var += self.e_var
#        neg_input = tf.concat((self.g_data, self.z), axis=1)
        self.d_neg_prob, self.d_var = self.discriminator(self.g_data)
#        pos_input = tf.concat((self.data, self.z_enc), axis=1)
        self.d_pos_prob, _ = self.discriminator(self.data, reuse=True)
        
        self.dz_neg_prob, self.d_var_z = self.discriminator_z(self.z)
        self.dz_pos_prob, _ = self.discriminator_z(self.z_enc, reuse=True)
        self.d_var += self.d_var_z
        
    def define_loss(self):
        self.d_loss = -tf.reduce_mean(tf.log(self.d_pos_prob + 1e-8) + tf.log(1. - self.d_neg_prob + 1e-8))
        self.d_loss += -tf.reduce_mean(tf.log(self.dz_pos_prob + 1e-8) + tf.log(1. - self.dz_neg_prob + 1e-8))
        self.g_loss = -tf.reduce_mean(tf.log(self.d_neg_prob + 1e-8) + tf.log(1. - self.d_pos_prob + 1e-8))
        data_mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.data, self.data_recons), 1))
#        z_mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.z, self.z_recons), 1))
#        z_kl = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - 2.0*self.logsigm - 1.0, 1))
        self.g_loss += data_mse
        
        
class fGAN(LSGAN):
    def __init__(self, data, hidden_num, z_dim, f_divergence='FKL'):
        self.f_divergence = f_divergence
        super(fGAN, self).__init__(data, hidden_num, z_dim)
        
    def define_loss(self):
        D_real = self.d_pos_prob
        D_fake = self.d_neg_prob
        if self.f_divergence == 'TV':
            """ Total Variation """
            D_loss = -(tf.reduce_mean(0.5 * tf.nn.tanh(D_real)) -
                    tf.reduce_mean(0.5 * tf.nn.tanh(D_fake)))
            G_loss = -tf.reduce_mean(0.5 * tf.nn.tanh(D_fake))
        elif self.f_divergence == 'FKL':
            """ Forward KL """
            D_loss = -(tf.reduce_mean(D_real) - tf.reduce_mean(tf.exp(D_fake - 1)))
            G_loss = -tf.reduce_mean(tf.exp(D_fake - 1))
        elif self.f_divergence == 'RKL':
            """ Reverse KL """
            D_loss = -(tf.reduce_mean(-tf.exp(D_real)) - tf.reduce_mean(-1 - D_fake))
            G_loss = -tf.reduce_mean(-1 - D_fake)
        elif self.f_divergence == 'PC':
            """ Pearson Chi-squared """
            D_loss = -(tf.reduce_mean(D_real) - tf.reduce_mean(0.25*D_fake**2 + D_fake))
            G_loss = -tf.reduce_mean(0.25*D_fake**2 + D_fake)
        elif self.f_divergence == 'SH':
            """ Squared Hellinger """
            D_loss = -(tf.reduce_mean(1 - tf.exp(D_real)) -
                    tf.reduce_mean((1 - tf.exp(D_fake)) / (tf.exp(D_fake))))
            G_loss = -tf.reduce_mean((1 - tf.exp(D_fake)) / (tf.exp(D_fake)))
        else:
            NotImplementedError('Not Implemented.')
        self.d_loss = D_loss
        self.g_loss = G_loss
        
        
        
            
            
            
            
            