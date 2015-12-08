# -*- coding: utf-8 -*-

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