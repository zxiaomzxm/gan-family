# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:54:22 2017

@author: zhaoxm
"""
import tensorflow as tf

class BaseSolver(object):
    def __init__(self, model, init_learning_rate):
        self.global_step = tf.Variable(0, trainable=False)
        self.init_learning_rate = init_learning_rate
        self.model = model
        self._train_op()
        
    def _train_op(self):
        self.learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step, 50000, 0.5, staircase=True)
        solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.d_solver = solver.minimize(self.model.d_loss, global_step=self.global_step, var_list=self.model.d_var)
        self.g_solver = solver.minimize(self.model.g_loss, global_step=self.global_step, var_list=self.model.g_var)
        
class WassersteinSolver(BaseSolver):
    def _train_op(self):
        self.learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step, 50000, 0.5, staircase=True)
        solver = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.d_solver = solver.minimize(self.model.d_loss, global_step=self.global_step, var_list=self.model.d_var)
        self.g_solver = solver.minimize(self.model.g_loss, global_step=self.global_step, var_list=self.model.g_var)
        self.clip_grad = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.model.d_var]