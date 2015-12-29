#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from glm.func.perceptron import Perceptron
from glm.func.tagger import Tagger
from glm.func.util import read_train_data, read_test_data, agg, read_prior, output_prior, ALL_LABEL
from glm.feature.features import *


class Argument:
    def __init__(self, params):
        self.param = dict((p.split(':')[0], p.split(':')[1]) for p in params)

    def get(self, key, default):
        return self.param.get(key, default)


# Previous word and current tag pair
preWordFeature = UniSurrFeature([-1])
# Next word and current tag pair
nextWordFeature = UniSurrFeature([1])

# pre pre word and current tag pair
prepreWordFeature = UniSurrFeature([-2])
# next next word and current tag pair
nextnextWordFeature = UniSurrFeature([2])

# pre pre pre word and current tag pair
preprepreWordFeature = UniSurrFeature([-3])
# next next next word and current tag pair
nextnextnextWordFeature = UniSurrFeature([3])

# [-1, 0] and current tag pair
precuWordFeature = UniSurrFeature([-1, 0])
# [0, 1] and current tag pair
cunextWordFeature = UniSurrFeature([0, 1])

# [-2, -1, 0] and current tag pair
preprecuWordFeature = UniSurrFeature([-2, -1, 0])
# [-1, 0, 1] and current tag pair
precunextWordFeature = UniSurrFeature([-1, 0, 1])
# [0, 1, 2] and current tag pair
cunextnextWordFeature = UniSurrFeature([0, 1, 2])


# Previous word and bigram tags
preWordBiFeature = BiSurrFeature([-1])
# Next word and bigram tags
nextWordBiFeature = BiSurrFeature([1])

# number feature
hasNumberFeature = ContainsFeature('0123456789$', 'NUM')
# special symbols
hasSpecialSymbolFeatures = \
    [ContainsFeature(*arg) for arg in [('.',), ('-',), ('&^%@#^!?[](){}*+/=', 'OTHER')]] + [hasNumberFeature]


#========= Feature template combination samples ==========
BASE = [TagFeature(True), BigramFeature()]
BASE_NOCASE = [TagFeature(False), BigramFeature()]
SUFFIX = lambda len1, len2, case=False : [SuffixFeature(leng, case) for leng in range(len1, len2 + 1)]
PREFIX = lambda len1, len2, case=False : [PrefixFeature(leng, case) for leng in range(len1, len2 + 1)]
SURR_UNI = [preWordFeature, nextWordFeature]
SURR_BI = [preWordBiFeature, nextWordBiFeature]

#========= Feature template used to train seg ==========
#features = [TagFeature(False), preWordFeature, nextWordFeature, prepreWordFeature, nextnextWordFeature, \
#    precuWordFeature, cunextWordFeature, preprecuWordFeature, precunextWordFeature, cunextnextWordFeature]

#========= Feature template used to train pos ==========
features = [TagFeature(False), preWordFeature, nextWordFeature] + SUFFIX(1, 1) + PREFIX(1, 1)


if __name__ == '__main__':
    args = Argument(sys.argv[2:])
    if sys.argv[1] == 'train':    
        iter_n = int(args.get('-it', 5))
        model_file = args.get('-m', 'glm/data/test.model')
        old_model_file = args.get('-om', None)
        train_file = args.get('-f', 'glm/data/train.dat')
        prior_file = args.get('-p', None)
        batch_num = int(args.get('-b', '2000'))
        # load all label
        ln = 0
        for (chunk, line) in read_train_data(train_file):
            ln += 1
        word_sta, tc, wc = agg(train_file)
        print >> sys.stdout, 'tag distribution', tc
        print >> sys.stdout, 'word count distribution', wc
        print >> sys.stdout, 'all label %s' % ALL_LABEL
        if prior_file != None:
            output_prior(prior_file, word_sta)
        p = Perceptron(train_file, features, model_file, old_model_file, word_sta if prior_file != None else {})
        p.train(iter_n, batch_num / float(ln))

    elif sys.argv[1] == 'tag':
        model_file = args.get('-m', 'glm/data/test.model')
        test_file = args.get('-f', 'glm/data/test.mini.dat')
        test_output_file = args.get('-o', 'glm/data/test.out')
        prior_file = args.get('-p', None)
        # load all label
        for l in read_test_data(test_file):
            pass
        p = Tagger(test_file, features, model_file, test_output_file, read_prior(prior_file) if prior_file else {})
        p.tag()
