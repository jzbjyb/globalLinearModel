# -*- coding: utf-8 -*-

import sys
from model import Model
from util import read_test_data, ALL_LABEL

class Tagger:
    def __init__(self, test_data_file, feature_template_list, model_file, test_output_file):
        self.test_data_file = test_data_file
        self.model = Model(feature_template_list, list(ALL_LABEL), model_file)
        self.test_output_file = test_output_file


    def tag(self):
        '''
        decode a sequence
        '''
        out_file = open(self.test_output_file, 'w')
        li = 0
        correct = [0, 0]
        for (chunk, line) in read_test_data(self.test_data_file):
            observe_data = [w[0] for w in line]            
            infer_label = self.model.viterbi(observe_data)
            if len(line) > 0 and len(line[0]) > 1:
                ideal_lable = [w[1] for w in line]
                for i in xrange(len(ideal_lable)):
                    correct[0] += 1 if ideal_lable[i] == infer_label[i] else 0
                    correct[1] += 1
            for (word, label) in zip(observe_data, infer_label):
                print >> out_file, word + '\t' + label
            #print >> sys.stdout, observe_data
            li += 1
            print >> sys.stdout, li
            print >> out_file
        print >> sys.stdout, float(correct[0]) / correct[1]