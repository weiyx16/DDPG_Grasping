# -*- coding: utf-8 -*-
#Reference:
#https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py

import sys
import os
import random
import logging
import numpy as np

from util.utils import *

class ReplayMemory:
    """
        -- Memory storage
    """
    def __init__(self, config):

        self.cnn_format = config.cnn_format
        self.inChannel = config.inChannel
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float16)
        self.screens = np.empty((self.memory_size, self.inChannel, config.screen_height // 4, config.screen_width // 4), dtype = np.float16)
        self.terminals = np.empty(self.memory_size, dtype = np.bool) # end or not
        self.history_length = config.history_length
        self.dims = (config.screen_height // 4, config.screen_width // 4)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length*self.inChannel) + self.dims, dtype = np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length*self.inChannel) + self.dims, dtype = np.float16)

    def add(self, screen, reward, action, terminal):
        # assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1) # from 0 -> memory_size then stop at the memory_size
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        """
            -- According to the index, get the corr history back
            i.e. corr_index and its history(history_length-1)
        """
        assert self.count > 0, " [!] replay memory is empty, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            hist_screens = self.screens[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            # 返回的是对应index的history_length长度的screen，直接压到channel里了
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            hist_screens = self.screens[indexes, ...]
        # hist_screens is history_length * inChannel * height * width (4-D tensor)
        # **NB** We need to covert to (history_length * inChannel) * h * w (3-D tensor)
        stacked_scr = np.reshape(hist_screens, (self.history_length*self.inChannel, self.dims[0], self.dims[1]))
        return stacked_scr

    def sample(self):
        """
            -- Sample from the memory (with index around history)
        """
        # memory must include poststate, prestate and ***history***
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over 
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state! (index can be terminal but it's precursor can't be)
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break     
            # NB! having index first is fastest in C-order matrices
            # Notice the index - 1 is saving the last screen
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        if self.cnn_format == 'NHWC':
            return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
                rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
        else:
            return self.prestates, actions, rewards, self.poststates, terminals