"""
    -- Deep Deterministic Policy Gradient Algorithm
    -- Take in RGBD-image(256*256) as input
    # Reference: https://github.com/stevenpjg/ddpg-aigym
"""
import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from functools import reduce

from .base import BaseModel
from .actor_net import ActorNet
from .critic_net import CriticNet
from .history import History
from .replay_memory import ReplayMemory
from tf_gradient_inverter import gradient_inverter

class Agent():
    def __init__(self, config, environment):
        # Load configuration from config.py
        self.config = config.list_all_member()
        for attr in self.config:
            if attr.startswith('__'):
                continue
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, self.config[attr])
        self.config = config
        self.env = environment
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config)

        self.critic_net = CriticNet(self.config)
        self.actor_net = ActorNet(self.config)

        action_bounds = [self.action_max, self.action_min]
        self.grad_inv = gradient_inverter(action_bounds)

    def train(self):
        """
            -- Training Process
        """
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_act_reward = 0
        terminal_times = 0
        ep_rewards, actions = [], []

        self.start_step = self.critic_net.step_cur()
        screen, reward, action, terminal = self.env.new_scene()
        
        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(self.start_step, self.max_step), ncols=70, initial=self.start_step):
            
            if self.step == self.learn_start + self.start_step:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. action from actor (notice introducing random noise)
            action = self.predict(self.history.get())
            # 2. act
            screen, reward, terminal = self.env.act(action, if_train=True)
            # 3. observe & store
            self.observe(screen, reward, action, terminal)
            # 4. learn
            self.learn()

            """
            # TODO: 2019.3.30
            # left things:
            # 1. definition of the terminal of the scence
            # 2. action definition
            # 3. what to print out and what to inject to summary
            # 4. how to restore and save both two network
            # 5. the replay memory function need to be adapted to our new model
            # 6. about the sess.run and .eval()
            # 7. main function
            # 8. debug with simulation 
            """
            # 注意 ep_reward属于在每次simulation里的总和
            # 把每次simulation得到总reward存成list放在ep_rewards里，在test_step来的时候存下来
            # 而 total_reward属于在test_step里的总和
            actions.append(action)
            total_reward += reward

            if terminal:
                terminal_times += 1
                if terminal_times >= 5:
                    # 注意！这个代码里训练部分没有epoch的概念，每次结束后，接着之前的step / memory 继续去尝试得到新的场景
                    # 所以才会出现memory中间有terminal的情况，在history_length周围有重新开始一次仿真的话，就不get这个sample
                    # 加上epoch（本质就是场景重新开始）也不能让step重置，也就是说初始的learn_start的step满足之后，就不会再回到learn_start的懵懂阶段了
                    # command = input('\n >> Continue ')
                    screen, reward, action, terminal = self.env.new_scene()
                    self.history.add(screen)
                    num_game += 1
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.
                    terminal_times = 0
                else:
                    # 移除已经独立的物体，而不改变剩下场景，可以让场景重复利用
                    screen, reward, action, terminal = self.env.new_scene(terminal_times)
                    self.history.add(screen)
                    num_game += 1
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.
            else:
                ep_reward += reward

            if self.step >= (self.learn_start + self.start_step):
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step # get in each action，所以在所有action数目里平均
                    avg_loss = self.total_loss / self.update_count # get total_loss in each mini-batch optimizer，所以在minibatch跑过的次数里平均
                    avg_q = self.total_q / self.update_count # get in each mini-batch optimizer

                    try:
                        # 在simulation的次数上平均
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print('''\n ----------------
                             \n [#] avg_act_r: %.4f, avg_l: %.6f, avg_q: %3.6f
                             \n ----------------
                             \n [#] avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d ''' \
                            % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

                    if max_avg_act_reward * 0.8 <= avg_reward:
                        # test之后得到一个比较好的结果，把这个model存下来
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        self.save_model(self.step + 1)
                        max_avg_act_reward = max(max_avg_act_reward, avg_reward)
                        
                    print('\n [#] Up-to-now, the max action reward is %.4f \n --------------- ' %(max_avg_act_reward))
                    
                    self.inject_summary({
                        'average.reward': avg_reward,
                        'average.loss': avg_loss,
                        'average.q': avg_q,
                        'episode.max reward': max_ep_reward,
                        'episode.min reward': min_ep_reward,
                        'episode.avg reward': avg_ep_reward,
                        'episode.num of game': num_game,
                        'episode.rewards': ep_rewards,
                        'episode.actions': actions,
                        'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),}
                        , self.step)
                    
                    # 注意这些信息都是每个test_step的轮回里进行存取读出的
                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

                    # force to renew the scene each test time to avoid the dead-loop
                    screen, reward, action, terminal = self.env.new_scene()
                    self.history.add(screen)
                    terminal_times = 0
    
    def evaluate_actor(self, state_batch):
        """
            Return action output from actor network
        """
        return self.actor_net.evaluate_actor(state_batch)
    
    def predict(self, state_batch, test_ep=None):
        """
            -- According to the estimation result -> get the prediction action (or exploration instead)
        """
        # TODO: How to add random strategy????

        action = self.evaluate_actor(state_batch)
        action = action[0]

        ep = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end)
                * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            # Exploration
            action = random.randrange(0, self.action_num) 

        return action

    def observe(self, screen, reward, action, terminal):
        """
            Add the action result into history(used to get the current result(next state) by the action)
            -- Notice the history is not used for mini-batch training!!
        """
        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

    def learn(self):
        """
            Learn from the memory storage every train_frequency (mini-batch loss GD)
            (according to the DDPG standard algorithm, set frequency = 1)
            and update target network's weights after learning (because we use soft update (tau) here)
        """
        if self.step > (self.learn_start + self.start_step): # in case of load model and retrain
            if self.step % self.train_frequency == 0:
                self.learning_mini_batch()
                self.critic_net.update_target_critic_network()
                self.actor_net.update_target_actor_network()

    def learning_mini_batch(self):
        """
            Mini batch GD from memory storage
            Notice the update of Critic anc Actor Network independently
        """
        if self.memory.count < self.history_length:
            return
        else:
            state_t, action, reward, state_t_plus_1, terminal = self.memory.sample()
        
        action_t_plus_1 = self.actor_net.evaluate_target_actor(state_t_plus_1)
        q_t_plus_1 = self.critic_net.evaluate_target_critic(state_t_plus_1, action_t_plus_1)

        # if terminal -> then reward
        # if not -> reward + decay * max (q)
        target_q_batch = []
        for i in range(self.batch_size):
            if terminal[i]:
                target_q_batch.append(reward[i])
            else:
                target_q_batch.append(reward[i] + self.discount*q_t_plus_1[i][0]) # what's 0 for?
        target_q_batch = np.array(target_q_batch)
        target_q_batch = np.reshape(target_q_batch, [len(target_q_batch), 1])
        '''or
        terminal = np.array(terminal) + 0.
        target_q_batch = (1. - terminal) * self.discount * q_t_plus_1 + reward
        '''
        # Q value update (the same to DQN but it need to get action_next from actor network first)
        _, q_t, loss_t = self.critic_net.train_critic(state_t, action, target_q_batch)

        # Policy Gradient (actor network update)
        action_t = self.evaluate_actor(state_t)
        if self.is_grad_inverter:
            Q_grdients_action = self.critic_net.compute_Q_grdients_action(state_t, action_t)
            Q_grdients_action = self.grad_inv.invert(Q_grdients_action, action_t)
        else:
            Q_grdients_action = self.critic_net.compute_Q_grdients_action(state_t, action_t)[0] # what's 0 for?
        self.actor_net.train_actor(state_t, Q_grdients_action)

        self.total_loss += loss_t
        self.total_q += q_t
        self.update_count += 1 # 记录优化次数