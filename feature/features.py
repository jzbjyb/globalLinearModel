# -*- coding: utf-8 -*-

from ..func.util import Label

class UnimplementedException(Exception): pass

class Feature:
    case = True 
    def __init__(self, name, case = None):
        self.name = name
        self.case = case
    
    @staticmethod
    def setAllCase(case):
        "set classwide case option. If self.case isn't set, use Feature.case"
        Feature.case = case


    def setCase(self, case):
        "Set to None to use Feature.case classwide setting"
        self.case = case
        return self


    def getCase(self):
        return Feature.case if self.case == None else self.case
   

    def _hash_str(self, *values):
        hashstr = self.name
        for val in values:
            hashstr += ':' + (str(val) if self.getCase() else str(val).lower())
        return hashstr
    

    def __call__(self, tag_seq, line, i):
        '''
        Any feature in the bigram global linear tagger will be defined by 
        [tag sequence], [array_of_sentence_words], i
        for bigram, the tag sequence would be [t[i-1], t[i]]
        note: index i starts from 0
        Should return a unique string as the key for hashing.
        Normally the string has format NAME:VALUE:VALUE
        '''
        raise UnimplementedException('Undefined abstract method')
    

    def __str__(self):
        '''
        * means the upper/lower case is ignored
        '''
        return ('' if self.getCase() else '*') + self.name


class TagFeature(Feature):
    '''
    <word, tag>
    '''
    def __init__(self, case=None):
        Feature.__init__(self, 'TAG', case)
    

    def __call__(self, tag_seq, line, i):
        return self._hash_str(line[i], tag_seq[-1])


class BigramFeature(Feature):
    '''
    <tag[-1], tag[0]>
    '''
    def __init__(self):
        Feature.__init__(self, 'BIGRAM', case=True)


    def __call__(self, tag_seq, line, i):
        return self._hash_str(*tag_seq[-2:])


class SuffixFeature(Feature):
    '''
    <suffix, tag>
    '''
    def __init__(self, suffixLen, case=None):
        Feature.__init__(self, 'SUFFIX', case)
        self.leng = suffixLen


    def __call__(self, tag_seq, line, i):
        return self._hash_str(line[i][-self.leng:], tag_seq[-1])
    

    def __str__(self):
        return Feature.__str__(self) + str(self.leng)

    
class PrefixFeature(Feature):
    '''
    <prefix, tag>
    '''
    def __init__(self, prefixLen, case=None):
        Feature.__init__(self, 'PREFIX', case)
        self.leng = prefixLen


    def __call__(self, tag_seq, line, i):
        return self._hash_str(line[i][:self.leng], tag_seq[-1])


    def __str__(self):
        return Feature.__str__(self) + str(self.leng)


class SurrFeature(Feature):
    '''
    Surrounding words and n-gram tag pair
    ngram=1 to consider only the current tag
    ngram=2 for the full bigram history
    indices: [-1, 0, 2] where 0 is the current position
    <indices, surr words, tags>
    '''
    def __init__(self, indices, ngram, case=None):
        Feature.__init__(self, 'SURR', case)
        self.ngram = ngram
        self.indices = indices
    

    def __call__(self, tag_seq, line, i):
        wordList = [Label.START if i + shift < 0 \
                    else (Label.END if i + shift >= len(line)\
                          else line[i + shift]) for shift in self.indices]
        # remember to hash indices
        return self._hash_str(*(self.indices + wordList + tag_seq[-self.ngram:]))


    def __str__(self):
        return Feature.__str__(self) + str(self.indices) + 'N' + str(self.ngram)
    

class UniSurrFeature(SurrFeature):
    '''
    <indices, surr words, tag>
    '''
    def __init__(self, indices, case=None):
        SurrFeature.__init__(self, indices, 1, case)
    def __str__(self):
        return SurrFeature.__str__(self)[:-2] + 'Uni'


class BiSurrFeature(SurrFeature):
    '''
    <indices, surr words, [tag[-1], tag[0]]>
    '''
    def __init__(self, indices, case=None):
        SurrFeature.__init__(self, indices, 2, case)
    def __str__(self):
        return SurrFeature.__str__(self)[:-2] + 'Bi'
        
        
class ContainsFeature(Feature):
    '''
    Contains one of the strings in [containList]
    containerName: e.g. NUM to denote a class of spelling features
    if not defined, we simply concat containList by '|' to construct a name
    if 'dual' set to True, updates both Contains and NotContains
    The feature would look like: 'CONTAIN:NUM:y:NOUN'
    the third field, 'y' for does contain and 'n' for doesn't contain
    if 'dual' set to False, only updates Contains, and NotContains would yield an emptry str
    Experiment shows that dual set to True gives better accuracy
    Note that this feature's default case is True, which will override the classwide setting
    '''
    def __init__(self, containList, containerName=None, dual=True, case=True):
        Feature.__init__(self, 'HAS', case)
        self.tokens = set(containList) if case else {token.lower() for token in containList}
        # create a meta-name
        self.name += ':' + (containerName or '|'.join(containList))
        self.dual = dual
    

    def __call__(self, tag_seq, line, i):
        word = line[i] if self.case else line[i].lower()
        found = len(set(word) & self.tokens) > 0
        if self.dual: return self._hash_str(['n', 'y'][found], tag_seq[-1])
        else: return self._hash_str(tag_seq[-1]) if found else ''


class ContainsCapital(ContainsFeature):
    def __init__(self):
        ContainsFeature.__init__(self, [chr(c) for c in range(ord('A'), ord('Z')+1)], 'CAP', dual=False, case=True)
    

    def __call__(self, tag_seq, line, i):
        # don't consider the capitalization at a sentence start
        return ContainsFeature.__call__(self, tag_seq, line, i) if i > 0 else ''


def features2str(feature_list):
    return '{' + ', '.join(feature_list) + '}'