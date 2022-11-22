import numpy as np
import math
from Simulator_Config import BfpConfig


class fp_container():
    def __init__(self):
        self.values = np.zeros(BfpConfig.group_size)
        self.group_size = BfpConfig.group_size
    def insert_value(self, index, value):
        if(index >= self.group_size or index < 0):
            print("error: index out of group_size")
        self.values[index] = value
        #print(self.values[index], " ",end='')
    def insert(self, container):
        for i in range(self.group_size):
            self.values[i] = container.get_value(i)
    def get_value(self, index):
        return self.values[index]

    