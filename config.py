class DQNConfig(object):

    def __init__(self, FLAGS):

        # ----------- Agent Params
        self.scale = 50
        # self.display = False
        self.tau = 0.01
        self.max_step = 400 * self.scale
        self.memory_size = 100 * self.scale

        self.batch_size = 8
        self.cnn_format = 'NCHW'
        self.discount = 0.9 # epsilon in RL (decay index)
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.001
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 4 * self.scale
        self.is_grad_inverter = True

        self.ep_end = 0.20
        self.ep_start = 1.
        self.ep_end_t = 2. * self.memory_size # encourage some new actions

        self.history_length = 4
        self.train_frequency = 4
        self.learn_start = 4. * self.scale

        self.test_step = 4 * self.scale

        # ----------- Environment Params
        self.env_name = 'Act_Perc'
        self.action_num = 2
        self.Lua_PATH = r'../affordance_model/infer.lua'
        # In furthur case you can just use local infomation with 64*64 or 128*128 pixel.
        self.screen_width  = 128
        self.screen_height = 128
        self.scene_num = 24
        self.test_scene_num = 6
        self.max_reward = 1.
        self.min_reward = -1.

        # ----------- Model Params
        self.inChannel = 4 # RGBD
        # self.action_repeat = 4
        self.critic_ckpt_dir = r'./ddpg/checkpoint_critic'
        self.actor_ckpt_dir = r'./ddpg/checkpoint_actor'
        self.model_dir = r'./ddpg/model'
        self.is_train = True
        self.is_sim = True
        self.end_metric = 0.85
        
        if FLAGS.use_gpu == False:
            self.cnn_format = 'NHWC'
        else:
            self.cnn_format = 'NCHW'

        if FLAGS.is_train == False:
            self.is_train = False

        if FLAGS.is_sim == False:
            self.is_sim = False
            
    def list_all_member(self):
        params = {}
        for name,value in vars(self).items():
            params[name] = value
        return params
