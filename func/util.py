# -*- coding: utf-8 -*-

from collections import defaultdict
import sys
import json

class Label(set):
    START = '*'
    END = '&'
    def __init__(self, lable_list = []):
        set.__init__(self, lable_list)


ALL_LABEL = Label()


def read_test_data(filename):
    '''
    yeild test data one "sentence" a time
    '''
    chunk = ''
    terms = []
    with open(filename) as file:
        for l in file:
            if not l.strip():
                yield chunk, terms
                chunk = ''
                terms = []
                continue
            chunk += l.strip()
            ls = l.strip().split('\t')
            if len(ls) > 1:
                ALL_LABEL.add(ls[-1])
                terms.append((ls[0], ls[1]))
            else:
                terms.append((ls[0]))



def read_train_data(filename):
    '''
    yeild train data one "sentence" a time
    '''
    chunk = ''
    terms = []
    with open(filename) as file:
        for l in file:
            if not l.strip():
                yield chunk, terms
                chunk = ''
                terms = []
                continue
            chunk += l.strip()
            ls = l.strip().split('\t')
            ALL_LABEL.add(ls[1])
            terms.append((ls[0], ls[1]))


def read_prior(filename):
    with open(filename, 'r') as file:
        prior = json.load(file)
    return prior

def output_prior(filename, prior):
    with open(filename, 'w') as file:
        file.write(json.dumps(prior))


TOTAL_TAG_STATISTIC = [1, 2, 3, 4, 5, 6, 7, 8]
TOTAL_WORD_STATISTIC = [1, 5, 25, 125, 625, 3125]

def agg(filename):
    result = defaultdict(lambda: defaultdict(lambda: 0))
    for (chunk, line) in read_train_data(filename):
        for w in line:
            result[w[0]][w[1]] += 1
    total_tc = []
    total_wc = []
    for w in result:
        wc = sum([result[w][t] for t in result[w]])
        tc = len(result[w])
        total_tc.append(tc)
        total_wc.append(wc)
    total_tc = sorted(total_tc)
    total_wc = sorted(total_wc)
    total_tc_result = []
    total_wc_result = []
    s = 0
    ub = -1
    for n in total_tc:
        if len(total_tc_result) == 0 or ub < n:
            total_tc_result.append(0)
        while s < len(TOTAL_TAG_STATISTIC) and TOTAL_TAG_STATISTIC[s] < n:
            s += 1
        ub = TOTAL_TAG_STATISTIC[s] if s < len(TOTAL_TAG_STATISTIC) else sys.maxint
        total_tc_result[-1] += 1
    s = 0
    ub = -1
    for n in total_wc:
        if len(total_wc_result) == 0 or ub < n:
            total_wc_result.append(0)
        while s < len(TOTAL_WORD_STATISTIC) and TOTAL_WORD_STATISTIC[s] < n:
            s += 1
        ub = TOTAL_WORD_STATISTIC[s] if s < len(TOTAL_WORD_STATISTIC) else sys.maxint
        total_wc_result[-1] += 1

    return result, \
        [(e, n) for e, n in zip(TOTAL_TAG_STATISTIC + [sys.maxint], total_tc_result)], \
        [(e, n) for e, n in zip(TOTAL_WORD_STATISTIC + [sys.maxint], total_wc_result)]