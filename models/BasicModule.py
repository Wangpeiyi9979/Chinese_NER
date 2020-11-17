# -*- coding: utf-8 -*-

import torch
import time

class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))  # model name

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        try:
            self.load_state_dict(torch.load(path))
        except:
            device = torch.device('cpu')
            self.load_state_dict(torch.load(path,  map_location=device))

    def save(self, name):
        torch.save(self.state_dict(), name)
        return name
