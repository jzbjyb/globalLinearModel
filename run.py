#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from glm.func.perceptron import Perceptron
from glm.func.tagger import Tagger
from glm.func.util import read_train_data, read_test_data, ALL_LABEL
from glm.feature.features import *
 
class Argument:
    def __init__(self, params):
        self.param = dict((p.split(':')[0], p.split(':')[1]) for p in params)

    def get(self, key, default):
        return self.param.get(key, default)

def gen_features_cmd(args):
    '''
    format: NAME:opt1,opt2
    Default: case sensitive. cf to set to false, must be the last opt
    '''
    features = []
    for token in args:
        if ':' in token: # name with option
            toks = token.split(':')
            name = toks[0]
            opts = toks[1].split(',')
            if opts[-1] == 'cf':
                case = False
                opts = opts[:-1]
            else:   case = None
        else:
            name = token
            opts = None
            case = None
        if name == 'tag':
            features.append(TagFeature(case=case))
        elif name == 'bigram':
            features.append(BigramFeature())
        elif name == 'suffix':
            # Default 1 to 3
            if not opts: opts = [1, 3]
            else: opts = [int(opt) for opt in opts]
            features += [SuffixFeature(l, case=case) for l in range(opts[0], opts[1]+1)]
        elif name == 'prefix':
            # Default 2 to 3
            if not opts: opts = [2, 3]
            else: opts = [int(opt) for opt in opts]
            features += [PrefixFeature(l, case=case) for l in range(opts[0], opts[1]+1)]
        elif name == 'unisurr':
            # Specify the surrounding word indices in opts
            if not opts: opts = [-1]
            else: opts = [int(opt) for opt in opts]
            features.append(UniSurrFeature(opts, case=case))
        elif name == 'bisurr':
            # Specify the surrounding word indices in opts
            if not opts: opts = [-1]
            else: opts = [int(opt) for opt in opts]
            features.append(BiSurrFeature(opts, case=case))
        elif name == 'has':
            if not opts: features += hasSpecialSymbolFeatures
            else:
                for opt in opts:
                    if opt == 'num': # number feature
                        features.append(hasNumberFeature)
                    elif opt == 'all': # all special symbols
                        features += hasSpecialSymbolFeatures
                    else:
                        features.append(ContainsFeature(opt))
        else: raise UnimplementedException('Unsupported feature: ' + name)
    return features


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

features = [TagFeature(False), preWordFeature, nextWordFeature, prepreWordFeature, nextnextWordFeature, \
    precuWordFeature, cunextWordFeature, preprecuWordFeature, precunextWordFeature, cunextnextWordFeature]

if __name__ == '__main__':
    args = Argument(sys.argv[2:])
    if sys.argv[1] == 'train':    
        iter_n = int(args.get('-it', 5))
        model_file = args.get('-m', 'glm/data/test.model')
        train_file = args.get('-f', 'glm/data/train.dat')
        batch_num = int(args.get('-b', '2000'))
        # load all label
        ln = 0
        for l in read_train_data(train_file):
            ln += 1
            pass
        print >> sys.stdout, 'all label %s' % ALL_LABEL
        p = Perceptron(train_file, features, model_file)
        p.train(iter_n, batch_num / float(ln))

    elif sys.argv[1] == 'tag':
        model_file = args.get('-m', 'glm/data/test.model')
        test_file = args.get('-f', 'glm/data/test.mini.dat')
        test_output_file = args.get('-o', 'glm/data/test.out')
        # load all label
        for l in read_test_data(test_file):
            pass
        p = Tagger(test_file, features, model_file, test_output_file)
        p.tag()
