# -*- coding: utf-8 -*-

from collections import defaultdict
import sys
import random
import numpy as np
import time

from util import Label

class Model:
    def __init__(self, featrue_template_list, label_list, model_file = None):
        '''
        model "vector" is implemented as a dict with featrue name as key
        feature name is generate by feature_templale function with input in form of 
        [s_i-1, s_i, X, i]
        '''
        self.cache = defaultdict(lambda: {})
        self.model = defaultdict(lambda: 0)
        self.feature = featrue_template_list
        self.label_list = label_list
        self.timef = 0

        # if have, load the model from file
        if model_file:
            with open(model_file) as file:
                for tags in file:
                    tags = tags.split()
                    try:
                        self.model[tags[0]] = float(tags[1])
                    except IndexError:
                        continue


    def __getitem__(self, key):
        '''
        key is in form of [s_i-1, s_i, X, i]
        return w_T dot* f(S, X) = w_T dot* sum_i_f(s_i-1, s_i, X, i)
        '''
        return sum([self.model[f(*key)] for f in self.feature])


    def delta(self, key, value):
        for f in self.feature:
            hk = f(*key)
            if hk:
                self.model[hk] += value


    def update(self, observe_data, ideal_label, infer_label):
        '''
        update model weight vector
        '''
        sample_num = len(observe_data)
        ideal_label = [Label.START] + ideal_label
        infer_label = [Label.START] + infer_label
        for i in xrange(1, sample_num + 1):
            if ideal_label[i-1] != infer_label[i-1] or ideal_label[i] != infer_label[i]:
                self.delta([[ideal_label[i-1], ideal_label[i]], observe_data, i-1], 1)
                self.delta([[infer_label[i-1], infer_label[i]], observe_data, i-1], -1)


    def prim(self, observe_data):

        sample_num = len(observe_data)
        label_num = len(self.label_list)
        graph = [[0 for j in xrange(sample_num)] for i in xrange(sample_num)]
        for i in xrange(sample_num):
            for j in xrange(i+1, sample_num):
                if i == j: continue
                [self[[i, observe_data, i, j]] for l in xrange(label_num)]


    '''
    prior is in the form:
    word -> [possible tag1, possible tag2, ...]
    '''
    def viterbi(self, observe_data, prior = {}):
        def argmax(ls): return max(ls, key = lambda x: x[1])
        sample_num = len(observe_data)
        label_num = len(self.label_list)
        '''
        metrix[i][j] means the max "probability" when the i th label is self.label_list[j]
        '''
        metrix = [[0 for l in xrange(label_num)] for i in xrange(sample_num)]
        #metrix = np.zeros(shape=(sample_num, label_num), dtype=np.int)
        '''
        back[i][j] mean the best previous label when the i th label is self.label_list[j]
        '''
        back = [[0 for l in xrange(label_num)] for i in xrange(sample_num)]
        #back = np.zeros(shape=(sample_num, label_num), dtype=np.int)

        # init metrix[0][0:len(self.label_list)]
        metrix[0] = [self[[[Label.START, self.label_list[l]], observe_data, 0]] \
            #if not prior.has_key(observe_data[0]) or prior[observe_data[0]].has_key(self.label_list[l]) \
            #else 0 \
            for l in xrange(label_num)]
        
        start = time.clock()
        # dynimic programming
        for i in xrange(1, sample_num):
            for l in xrange(label_num):
                #if prior.has_key(observe_data[i]) and not prior[observe_data[i]].has_key(self.label_list[l]):
                #    back[i][l] = -1
                #    metrix[i][l] = 0
                #else:
                max_pre = -1
                max_score = -sys.maxint
                cu_ws = self[[[Label.START, self.label_list[l]], observe_data, i]]
                for lj in xrange(label_num):
                    sc = metrix[i-1][lj] + cu_ws
                    if sc > max_score:
                        max_score = sc
                        max_pre = lj
                    
                    #back[i][l], metrix[i][l] = argmax([ \
                    #    (lj, metrix[i-1][lj] + self[[[self.label_list[lj], self.label_list[l]], observe_data, i]]) \
                        #if not prior.has_key(observe_data[i-1]) or prior[observe_data[i-1]].has_key(self.label_list[lj]) \
                        #else (-1, -sys.maxint) \
                    #    for lj in xrange(label_num)])
                
                metrix[i][l] = max_score
                back[i][l] = max_pre
        end = time.clock()
        self.timef += end - start

        # back infer
        infer_label = [argmax([(j, metrix[sample_num-1][j]) for j in xrange(label_num)])[0]]
        for i in xrange(sample_num - 1):
            infer_label.append(back[sample_num - i - 1][infer_label[-1]])
        infer_label.reverse()
        return [self.label_list[l] for l in infer_label]


    def __str__(self):
        '''
        used to output the model to file
        '''
        ml = []
        for hashstr in self.model:
            score = self.model[hashstr]
            if score != 0:
                ml.append(hashstr + '\t' + str(score))
        return '\n'.join(ml)
