# -*- coding: utf-8 -*-

import sys
import time
import random
from model import Model
from util import read_train_data, ALL_LABEL

class Perceptron:
    def __init__(self, train_data_file, feature_template_list, model_file):
        self.train_data_file = train_data_file
        self.model = Model(feature_template_list, list(ALL_LABEL))
        self.model_file = model_file

    def train(self, iteration, keep):
        '''
        perceptron train algorithm
        '''
        for it in xrange(iteration):
            viterbi_time = 0
            update_time = 0
            on = 0
            ln = 0
            label_len = len(ALL_LABEL) ** 2
            same = [0, 0]
            print >> sys.stderr, 'perceptron iteration', it + 1
            for (chunk, line) in read_train_data(self.train_data_file):
                ln += 1
                if ln % 1000 == 0:
                    print viterbi_time, update_time
                if random.random() > keep:
                    continue
                start = time.clock()
                observe_data = [w[0] for w in line]
                ideal_lable = [w[1] for w in line]
                infer_label = self.model.viterbi(observe_data)
                ss = 0
                for li in xrange(len(ideal_lable)): ss += 1 if ideal_lable[li] == infer_label[li] else 0
                same[0] += float(ss) / len(ideal_lable)
                same[1] += 1
                on += len(observe_data) * label_len
                end = time.clock()
                viterbi_time += end - start
                start = time.clock()
                self.model.update(observe_data, ideal_lable, infer_label)
                end = time.clock()
                update_time += end - start                
            print viterbi_time, update_time, on, same[0] / same[1]
        print >> open(self.model_file, 'w'), self.model