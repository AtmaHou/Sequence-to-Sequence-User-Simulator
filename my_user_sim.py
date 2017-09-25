# coding:utf-8
# Author: Atma Hou
# Contact: ythou@ir.hit.edu.cn
# Reference: Layla et al. 2016. A Sequence-to-Sequence Model for User Simulation in Spoken Dialogue Systems.
from dataset_walker import dataset_walker
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, GRU, Embedding, TimeDistributed
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
import itertools


# config for running
data_folder = 'data'


class UserSim:
    def __init__(self, dataset, act_config=None):
        self.data = dataset
        if act_config:
            self.informable = act_config['informable']
            self.requestable = act_config['requestable']
            self.machine_acts = act_config['machine_acts']
            self.user_acts = act_config['user_acts']
        else:
            self.informable = ['area', 'food', 'name', 'pricerange']
            self.requestable = self.informable + ['addr', 'phone', 'postcode', 'signature']
            self.machine_acts = ['affirm', 'bye', 'canthear', 'confirm-domain', 'negate', 'repeat', 'reqmore',
                           'welcomemsg', 'canthelp', 'canthelp.missing_slot_value', 'canthelp.exception', 'expl-conf',
                           'impl-conf', 'inform',
                           'offer', 'request', 'select', 'welcomemsg']

            self.user_acts = ['ack', 'affirm', 'bye', 'hello', 'help', 'negate', 'null', 'repeat', 'reqalts',
                        'reqmore', 'restart', 'silence', 'thankyou', 'confirm', 'deny', 'inform', 'request']
        # Size means picking 1 user act for each decoding output
        self.input_len = len(self.user_acts)
        # Size is the same to the context vector
        self.output_len = 3 * len(self.informable) + len(self.requestable) + len(self.machine_acts)
        # All data



if __name__ == '__main__':
    dataset = dataset_walker("dstc2_dev", dataroot=data_folder, labels=True)
    my_user_sim = UserSim(dataset)
