# -*- coding: utf-8 -*-
"""
    -- This module is for critic network of DDPG
    -- From state and action -> Q value
"""

import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
from functools import reduce

from .base import BaseModel
from .ops import linear, conv2d

class CriticNet(BaseModel):
    def __init__(self, config):
        super(CriticNet, self).__init__(config)
        
        self.g=tf.Graph()

        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            self._build_critic()

    """
        ############ Build The Critic Network ############
    """
    def _build_critic(self):
        """
            Build the Critic network
        """
        print(' [*] Building Crtic Network')

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        linear_activation_fn = tf.nn.tanh

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder(tf.int32, None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)
        
        # Prediction Critic Network
        with tf.variable_scope('prediction'):
            if self.cnn_format == 'NHWC':
                self.s_t = tf.placeholder(tf.float32,
                    [None, self.screen_height , self.screen_width, self.inChannel*self.history_length], name='s_t')
            else:
                self.s_t = tf.placeholder(tf.float32,
                    [None, self.inChannel*self.history_length, self.screen_height, self.screen_width], name='s_t')
            
            self.action_t = tf.placeholder(tf.float32, [None, self.action_num], name = 'action_t')

            # s_t = None*128*128*16(RGBD*4 previous frames)
            # downsample_1
            self.w = {}
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
                64, [6, 6], [3, 3], initializer, activation_fn, self.cnn_format, name='_l1')
            # l1 = None*41*41*64
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='_l2')
            # l2 = None*19*19*64
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='_l3')
            # l3 = None*17*17*64

            shape = self.l3.get_shape().as_list()
            # 将输出沿着batch size那一层展开，为了后面可以接到全连接层里
            # dim of l3_flat = batch_size * (H*W*C)
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            # Dense Connection
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=linear_activation_fn, name='_l4')

            # Output
            self.l5, self.w['l5_state_w'], self.w['l5_action_w'], self.w['l5_b'] = linear(self.l4, 256, input_other=self.action_t, activation_fn=linear_activation_fn, name='_l5')
            
            # Q value for Current Action [shape = batchsize*1]
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l5, 1, activation_fn=None, name='q')
        print(' [*] Build Critic-Online Scope')

        # target network
        # The structure is the same with eval network
        with tf.variable_scope('target'):
            if self.cnn_format == 'NHWC':
                self.target_s_t = tf.placeholder(tf.float32,
                    [None, self.screen_height , self.screen_width, self.inChannel*self.history_length], name='target_s_t')
            else:
                self.target_s_t = tf.placeholder(tf.float32,
                    [None, self.inChannel*self.history_length, self.screen_height, self.screen_width], name='target_s_t')
            
            self.target_action_t = tf.placeholder(tf.float32, [None, self.action_num], name = 'target_action_t')

            # s_t = None*128*128*16(RGBD*4 previous frames)
            # downsample_1
            self.target_w = {}
            self.target_l1, self.target_w['l1_w'], self.target_w['l1_b'] = conv2d(self.target_s_t,
                64, [6, 6], [3, 3], initializer, activation_fn, self.cnn_format, name='target_l1')
            # l1 = None*41*41*64
            self.target_l2, self.target_w['l2_w'], self.target_w['l2_b'] = conv2d(self.target_l1,
                64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
            # l2 = None*19*19*64
            self.target_l3, self.target_w['l3_w'], self.target_w['l3_b'] = conv2d(self.target_l2,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')
            # l3 = None*17*17*64

            shape = self.target_l3.get_shape().as_list()
            # 将输出沿着batch size那一层展开，为了后面可以接到全连接层里
            # dim of l3_flat = batch_size * (H*W*C)
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            # Dense Connection
            self.target_l4, self.target_w['l4_w'], self.target_w['l4_b'] = linear(self.target_l3_flat, 512, activation_fn=linear_activation_fn, name='target_l4')

            # Output
            self.target_l5, self.target_w['l5_state_w'], self.target_w['l5_action_w'], self.target_w['l5_b'] = \
                linear(self.target_l4, 256, input_other=self.target_action_t, activation_fn=linear_activation_fn, name='target_l5')
            
            # Q value for Current Action [shape = batchsize*1]
            self.target_q, self.target_w['q_w'], self.target_w['q_b'] = linear(self.target_l5, 1, activation_fn=None, name='target_q')
        print(' [*] Build Critic-Target Scope')

        # Used to Set target network params from estimation network (let the t_w_input = w, then assign target_w with t_w_input)
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder(tf.float32, self.target_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])
        print(' [*] Build Critic Weights Transform Scope')

        # optimizer
        with tf.variable_scope('optimizer'):
            self.q_in = tf.placeholder(tf.float32, [None, 1], name = 'supervisor_q') # TODO: None or [None,1] & regularizer_loss?
            self.delta = self.q_in - self.q
            self.loss = tf.reduce_mean(tf.pow(self.delta, 2), name = 'loss') #self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder(tf.int64, None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                tf.train.exponential_decay(
                    self.learning_rate,
                    self.learning_rate_step,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay,
                    staircase=True))

            self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.9, epsilon=0.01).minimize(self.loss)
        print(' [*] Build Optimize Scope')

        with tf.variable_scope('action_gradients'):
            self.act_grad_v = tf.gradients(self.q, self.action_t)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])] # normalize along the batch_size

        # display all the params in the tfboard by summary
        with tf.variable_scope('summary'):
            # save every Mini_batch GD
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.scalar("%s/%s" % (self.env_name, tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

            self.writer = tf.summary.FileWriter('./ddpg/critic_logs', self.sess.graph)
        print(' [*] Build Critic Summary Scope')

        tf.global_variables_initializer().run()
        print(' [*] Initial All Critic Variables')
        self._saver = tf.train.Saver(list(self.w.values()) + [self.step_op], max_to_keep = 10, keep_checkpoint_every_n_hours=2)

        self.load_model(is_critic = True)
        self.update_target_critic_network(is_initial = True)

    def update_target_critic_network(self, is_initial = False):
        """
            Assign estimation network weights to target network. (not simultaneous)
            TODO: is that ok to use eval function here?
        """
        if is_initial:
            for name in self.w.keys():
                self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
        else:
            for name in self.w.keys():
                self.t_w_assign_op[name].eval({self.t_w_input[name]: self.tau*self.w[name].eval()+(1-self.tau)*self.target_w[name].eval()}) 
        print(' [*] Assign Weights from Prediction to Target')

    """
        ############ Train and Evaluation ############
    """
    def train_critic(self, state_batch, action_batch, target_q_batch):
        return self.sess.run([self.optim, self.q, self.loss], feed_dict={self.s_t: state_batch, self.action_t: action_batch, self.q_in: target_q_batch})

    def evaluate_target_critic(self, target_state_batch, target_action_batch):
        return self.sess.run(self.target_q, feed_dict={self.target_s_t:target_state_batch, self.target_action_t:target_action_batch})

    def compute_Q_grdients_action(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={self.s_t: state_batch, self.action_t: action_batch})

    def step_cur(self):
        return self.sess.run(self.step_op)