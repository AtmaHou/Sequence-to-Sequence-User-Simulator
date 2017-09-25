# coding:utf-8
"""
Author: Atma Hou
Contact: ythou@ir.hit.edu.cn
Reference: Layla et al. 2016. A Sequence-to-Sequence Model for User Simulation in Spoken Dialogue Systems.
"""

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
    def __init__(self, act_config=None):
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
        # input len means picking 1 user act for each decoding output
        self.input_len = len(self.user_acts)
        # output len is the same to the context vector
        self.output_len = 3 * len(self.informable) + len(self.requestable) + len(self.machine_acts)
        # All data
        self.x_all = []
        self.y_all = []

    def generate_training_data(self, dataset, context_turn_num=3):
        blank_turn_input = [0] * self.input_len
        for call in list(dataset):
            # Init the dialogue history with blank context vector to make sure the
            # input sequence of LSTM have same length
            dialogue_history = [blank_turn_input] * (context_turn_num - 1)
            constaint_vector = [0] * len(self.informable)
            request_vector = [0] * len(self.requestable)
            user_act_vector = [0] * len(self.user_acts)

            constraint_values = [''] * len(self.informable)
            context_history = []

            for turn, label in call:
                machine_act_vector = [0] * len(self.machine_acts)
                inconsistency_vector = [0] * 2 * len(self.informable)
                machine_mentioned = []

                # Build machine act vector
                for act in turn['output']['dialog-acts']:
                    act_string = act['act']
                    machine_act_vector[self.machine_acts.index(act_string)] = 1
                    # check misunderstanding
                    if act_string == 'inform':
                        for slot in act['slots']:
                            if slot[0] in self.informable:
                                machine_mentioned.append(slot[1])


if __name__ == '__main__':
    dataset = dataset_walker("dstc2_dev", dataroot=data_folder, labels=True)
    my_user_sim = UserSim()
