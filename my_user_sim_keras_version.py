# coding:utf-8
"""
Author: Atma Hou
Contact: ythou@ir.hit.edu.cn
Reference: Layla et al. 2016. A Sequence-to-Sequence Model for User Simulation in Spoken Dialogue Systems.
"""

from dataset_walker import dataset_walker
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, GRU, Embedding, TimeDistributed
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

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
        # input len is the same to the context vector
        self.input_len = 3 * len(self.informable) + len(self.requestable) + len(self.machine_acts)
        # output len means picking 1 user act for each decoding output
        self.output_len = len(self.user_acts)
        # offset is used to build inconsistency vector
        self.offset = len(self.informable)
        # All data
        self.x_all = []
        self.y_all = []
        self.goal_state_all = []  # {'constraint': [...], 'request':[...]}
        # Init model as sequential model
        self.model = Sequential()
        self.model_file = 'my_s2s_model.h5'

    def define_model(self, x_shape):
        self.model.add(LSTM(output_dim=self.output_len, input_shape=(x_shape[1], x_shape[2]), activation='relu',
                       return_sequences=True))
        self.model.add(GRU(output_dim=self.output_len, input_shape=(x_shape[1], x_shape[2]), activation='relu',
                      return_sequences=False))
        self.model.add(Dense(output_dim=self.output_len, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def generate_training_data(self, dataset, context_turn_num=3):
        blank_turn_input = [0] * self.input_len
        for call in list(dataset):
            # Init the dialogue history with blank context vector to make sure the
            # input sequence of LSTM have same length
            dialogue_history = [blank_turn_input] * (context_turn_num - 1)
            constraint_vector = [0] * len(self.informable)
            request_vector = [0] * len(self.requestable)
            user_act_vector = [0] * len(self.user_acts)
            constraint_values = [''] * len(self.informable)
            request_values = [''] * len(self.requestable)

            for turn, label in call:
                machine_act_vector = [0] * len(self.machine_acts)
                inconsistency_vector = [0] * 2 * len(self.informable)
                machine_mentioned = {
                    'slot': [],
                    'value': []
                }
                # Build machine act vector,and inconsistency first turn's inconsistency is empty
                for act in turn['output']['dialog-acts']:
                    act_string = act['act']
                    machine_act_vector[self.machine_acts.index(act_string)] = 1
                    # check misunderstanding
                    if act_string == 'inform':
                        for slot in act['slots']:
                            slot_name = slot[0]
                            slot_value = slot[1]
                            if slot_name in self.informable:
                                machine_mentioned['value'].append(slot_value)
                                machine_mentioned['slot'].append(slot_name)
                                # user's constraint value is conflict with system's proposal
                                if constraint_values[self.informable.index(slot_name)] != '' \
                                and slot_value != constraint_values[self.informable.index(slot_name)]:
                                    inconsistency_vector[self.offset + self.informable.index(slot_name)] = 1
                                    constraint_vector[self.informable.index(slot_name)] = 0
                # Build inconsistency vector & constraint vector
                for ind, c_value in enumerate(constraint_values):
                    # slot in user goal but not mentioned by agent
                    if c_value != '' and c_value != 'dontcare' and c_value not in machine_mentioned['value']:
                        inconsistency_vector[ind] = 1
                        constraint_vector[ind] = 1

                # Update constraint vector, remove informed slot
                for ind, c_value in enumerate(constraint_values):
                    if inconsistency_vector[ind] != 1 and inconsistency_vector[ind + self.offset] != 1 and c_value != '':
                        constraint_vector[ind] = 0

                # Update request vector, remove received slot
                for ind, r_value in enumerate(request_values):
                    if r_value != '' and r_value in machine_mentioned['slot']:
                        request_vector[ind] = 0

                # Build context/label vector, first turn's inconsistency is empty
                context_vector = machine_act_vector + inconsistency_vector + constraint_vector + request_vector
                self.y_all.append(user_act_vector)
                goal_state = {
                    'c_vector': constraint_vector,
                    'r_vector': request_vector
                }
                # Build request vector,
                for request in label['requested-slots']:
                    request_vector[self.requestable.index(request)] = 1
                    request_values[self.requestable.index(request)] = request

                # Build constraint vector
                for semantic in label['semantics']['json']:
                    if semantic['act'] == 'inform':
                        for slot in semantic['slots']:
                            if slot[0] in self.informable and slot[1] != 'dontcare':
                                constraint_values[self.informable.index(slot[0])] = slot[1]

                # Build one user act vector
                user_act_vector = [0] * len(self.user_acts)
                for sem in label['semantics']['json']:
                    user_act_vector[self.user_acts.index(sem['act'])] = 1

                dialogue_history.append(context_vector)
                self.x_all.append(dialogue_history[-1 * context_turn_num:])
                goal_state['c_value'] = constraint_values
                goal_state['r_value'] = request_vector
                self.goal_state_all.append(goal_state)

    def train(self, x_train, y_train, save_pm=True):
        self.define_model(x_train.shape)
        self.model.fit(x_train, y_train, batch_size=16, nb_epoch=10)
        if save_pm:
            self.model.save(self.model_file)

    def test(self, x_test, y_test):
        print 'Evaluate'
        # basic evaluation
        print self.model.evaluate(x_test, y_test, batch_size=100, verbose=1)
        # other evaluation
        predictions = self.model.predict(x_test)
        return predictions

    def train_and_test(self, training_percent=0.7, training=True, testing=True):
        train_set_size = int(training_percent * len(self.x_all))
        self.x_all = np.array(self.x_all)
        self.y_all = np.array(self.y_all)

        x_train = self.x_all[:train_set_size]
        y_train = self.y_all[:train_set_size]
        goal_state_train = self.goal_state_all[:train_set_size]
        x_test = self.x_all[train_set_size:]
        y_test = self.y_all[train_set_size:]
        goal_state_test = self.goal_state_all[train_set_size:]
        if not training and os.path.exists(self.model_file):
            self.model = load_model(self.model_file)
        else:
            self.train(x_train, y_train, save_pm=True)

        if testing:
            model_predictions = self.test(x_test, y_test)
            self.type_level_evaluate(model_predictions, y_test)
            self.fine_grained_evaluate(model_predictions, y_test, goal_state_test)

    def type_level_evaluate(self, model_predictions, y_test):
        # useless in-fact.....
        correct = 0
        predictions = []
        precision = 0.0
        recall = 0.0
        with open('type_level_eval_log.txt', 'w') as writer:
            for j in range(0, len(model_predictions)):
                predicted = (1 / (1 + np.exp(-np.array(model_predictions[j])))).tolist()
                actual = y_test[j].tolist()
                local_precision = 0.0
                local_recall = 0.0

                for i in range(0, len(predicted)):
                    if predicted[i] >= 0.58:
                        predicted[i] = 1
                    else:
                        predicted[i] = 0

                if predicted == actual:
                    correct += 1

                for pos in range(len(predicted)):
                    if predicted[pos] == 1:
                        if actual[pos] == 1:
                            local_precision += 1

                for pos in range(len(actual)):
                    if actual[pos] == 1:
                        if predicted[pos] == 1:
                            local_recall += 1
                precision_count = predicted.count(1)
                local_precision_avg = 0.0
                if precision_count != 0:
                    local_precision_avg += local_precision / precision_count

                recall_count = actual.count(1)
                local_recall_avg = 0.0
                if recall_count != 0:
                    local_recall_avg = local_recall / recall_count

                precision += local_precision_avg
                recall += local_recall_avg

                predictions.append(predicted)

                eval_str = '\n[%d]\npredicted: %s\nactual: %s\nLocal Precision: %f\nLocal Recall: %f\n' % (
                    j,
                    str(predicted),
                    str(actual),
                    local_precision_avg,
                    local_recall_avg,
                )
                writer.write(eval_str)

        print 'Accuracy: %f' % (correct * 1.0 / len(model_predictions))
        print 'Precision: %f' % (precision / len(model_predictions))
        print 'Recall: %f' % (recall / len(model_predictions))

    def fine_grained_evaluate(self, model_predictions, y_test, goal_state_test):
        print 'start fine grained test'
        correct = 0
        predictions = []
        precision = 0.0
        recall = 0.0
        with open('fine_grained_eval_log.txt', 'w') as writer:
            for j in range(0, len(model_predictions)):
                type_predicted = (1 / (1 + np.exp(-np.array(model_predictions[j])))).tolist()
                type_actual = y_test[j].tolist()
                goal_state = goal_state_test[j]

                local_precision = 0.0
                local_recall = 0.0

                for i in range(0, len(type_predicted)):
                    if type_predicted[i] >= 0.58:
                        type_predicted[i] = 1
                    else:
                        type_predicted[i] = 0

                if type_predicted == type_actual:
                    correct += 1

                # ========== transfer to fine-grained level ===========
                predicted = type_predicted[: -2]
                actual = type_actual[: -2]
                # deal with inform
                if type_predicted[-2]:  # inform act selected
                    predicted.extend(goal_state['c_vector'])
                else:
                    predicted.extend([0] * len(self.informable))
                inform_ans = [0] * len(self.informable)
                if type_actual[-2]:
                    for ind, c_value in enumerate(goal_state['c_value']):
                        if c_value != '':
                            inform_ans[ind] = 1
                actual.extend(inform_ans)
                # deal with req
                if type_predicted[-1]:  # from act selected
                    predicted.extend(goal_state['r_vector'])
                else:
                    predicted.extend([0] * len(self.requestable))
                request_ans = [0] * len(self.requestable)
                if type_actual[-1]:
                    for r_value in goal_state['r_value']:
                        if r_value:
                            request_ans[self.requestable.index(r_value)] = 1
                actual.extend(request_ans)
                print "==== debug ====", len(predicted), len(actual)
                # =========== start test ==============
                for pos in range(len(predicted)):
                    if predicted[pos] == 1:
                        if actual[pos] == 1:
                            local_precision += 1

                for pos in range(len(actual)):
                    if actual[pos] == 1:
                        if predicted[pos] == 1:
                            local_recall += 1
                precision_count = predicted.count(1)
                local_precision_avg = 0.0
                if precision_count != 0:
                    local_precision_avg += local_precision / precision_count

                recall_count = actual.count(1)
                local_recall_avg = 0.0
                if recall_count != 0:
                    local_recall_avg = local_recall / recall_count

                precision += local_precision_avg
                recall += local_recall_avg

                predictions.append(predicted)

                eval_str = '\n[%d]\npredicted: %s\nactual: %s\nLocal Precision: %f\nLocal Recall: %f\n' % (
                    j,
                    str(predicted),
                    str(actual),
                    local_precision_avg,
                    local_recall_avg,
                )
                writer.write(eval_str)

        print 'Accuracy: %f' % (correct * 1.0 / len(model_predictions))
        print 'Precision: %f' % (precision / len(model_predictions))
        print 'Recall: %f' % (recall / len(model_predictions))

if __name__ == '__main__':
    dataset = dataset_walker("dstc2_dev", dataroot=data_folder, labels=True)
    my_user_sim = UserSim()
    my_user_sim.generate_training_data(dataset, context_turn_num=3)
    my_user_sim.train_and_test(training_percent=0.7, training=True, testing=True)
