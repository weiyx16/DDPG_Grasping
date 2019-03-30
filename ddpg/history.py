# Reference:
# https://github.com/devsisters/DQN-tensorflow
import numpy as np

class History:
    def __init__(self, config):
        self.cnn_format = config.cnn_format

        history_length, self.inChannel, screen_height, screen_width = \
            config.history_length, config.inChannel, config.screen_height, config.screen_width

        self.history = np.zeros(
            [history_length*self.inChannel, screen_height, screen_width], dtype=np.float32)

    def add(self, screen):
        self.history[:-self.inChannel] = self.history[self.inChannel:] # 1,2,3 -> 0,1,2 (if history_length = 4)
        self.history[-self.inChannel:] = screen # 3 -> new

    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
            return np.transpose(self.history, (1, 2, 0)) # -> output H*W*(history_length*self.inChannel)
        else:
            return self.history
