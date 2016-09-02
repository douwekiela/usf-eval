'''
Code for loading and evaluating models given a certain metric
'''

import os, sys
import operator

import cPickle as pickle
import json

import numpy as np

from multiprocessing import Pool

from sims import cosine
import metrics

'''
The class accepts two types of input files:
 - A pickle or json file that contains a vector space of the format

    {'word1': np.array(),
     ...
     'word_n': np.array()}

 - A text file that contains a similarity matrix of the format

    word1 word2 sim_score
    ...
    word_n word_n sim_score

'''

def rankScores(w, scores):
    return w, sorted(scores, key=operator.itemgetter(1), reverse=True)

class ModelEval(object):
    def __init__(self, eval, fname, simFunc=cosine, verbose=True):
        self.eval = eval
        self.evalwords = list(set([w1 for w1 in eval] + [x[0] for w1 in eval for x in eval[w1]]))
        self.fname = fname
        self.simFunc = simFunc
        self.verbose = verbose

        # do some checks
        first_key = self.eval.keys()[0]
        assert isinstance(self.eval[first_key], list)
        assert isinstance(self.eval[first_key][0], list) or isinstance(self.eval[first_key][0], tuple)

        _, ext = os.path.splitext(fname)
        if ext in ['.pkl', '.json', '.words']:
            self.vspace = self.loadVectorSpace(fname, ext)
            self.sims = self.computeSimilarityMatrix(self.vspace, self.simFunc)
            self.ranks = self.rankOrder(self.sims)
            self.dumpRankedSimilarityMatrix(self.ranks, open(fname + '_sims.txt', 'w'))
        elif ext in ['.txt']:
            self.ranks = self.loadRankedSimilarityMatrix(fname)
        else:
            raise ValueError('Input file format not supported')

    '''
    Load vector space from a Pickle or JSON file

    Arguments:
        fname - file name
        ext - extension/file type
    Returns:
        vspace - a vector space dict
    '''
    def loadVectorSpace(self, fname, ext):
        if ext == '.json':
            vspace =json.load(open(fname))
        elif ext == '.pkl':
            vspace = pickle.load(open(fname), 'rb')
        elif ext == '.words':
            vspace = {}
            for line in open(fname):
                l = line.split()
                vspace[l[0]] = np.asarray([float(x) for x in l[1:]])
        return vspace

    '''
    Load ranked similarity matrix from a text file

    Arguments:
        fname - file name
    Returns:
        ranks - ranked similarity matrix
    '''
    def loadRankedSimilarityMatrix(self, fname):
        ranks = {}

        if self.verbose:
            print('Loading ranked similarity matrix')

        with open(fname) as f:
            for w1, w2, sim in [line.split() for line in f]:
                if w1 not in ranks:
                    ranks[w1] = []
                ranks[w1].append((w2, float(sim)))
        return ranks

    '''
    Dump ranked similarity matrix to a text file

    Arguments:
        ranks - ranked similarity matrix
        fw - writable file descriptor
    '''
    def dumpRankedSimilarityMatrix(self, ranks, fw):
        if self.verbose:
            print('Dumping similarity matrix')

        for w1 in ranks:
            fw.write("\n".join([" ".join([w1, w2, str(score)]) for (w2, score) in ranks[w1]]) + "\n")

    '''
    Compute similarity matrix from a vector space

    Arguments:
        vspace - vector space
        simFunc - similarity function that takes two vectors as arguments
    Returns:
        sims - similarity matrix
    '''
    def computeSimilarityMatrix(self, vspace, simFunc):
        sims = {}

        if self.verbose:
            print('Computing similarity matrix')

        for w1 in self.eval:
            if w1 not in vspace: continue
            v1 = vspace[w1]
            sims[w1] = {w2:simFunc(v1, vspace[w2]) for w2 in [w for w in self.evalwords if w in vspace and w != w1]}

        return sims

    '''
    Compute rank order of similarity matrix

    Arguments:
        sims - similarity matrix
    Returns:
        ranks - ranked matrix
    '''
    def rankOrder(self, sims):
        ranks = {}

        print sims
        if self.verbose:
            print('Ranking similarity matrix')

        p = Pool(16)
        for w1 in sims:
            def callback(result):
                idx, sorted_list = result
                print idx
                ranks[idx] = sorted_list
            p.apply_async(rankScores, (w1, sims[w1].items(), ), callback=callback)
        p.close()
        p.join()

        return ranks

    '''
    Run evaluation metric over rank order similarity matrix

    Arguments:
        ranks - ranked matrix
    Returns:
        score - averaged metric output
    '''
    def evalMetric(self, metric):
        score, n = 0, 0

        if self.verbose:
            print('Running evaluation')

        for w1 in self.eval:
            if w1 not in self.ranks: # skip non-existing
                continue

            # TODO: Implement desired metrics here
            if metric == 'precision_at_10':
                k = 10
                rs = [1 if w in [x[0] for x in self.eval[w1]] else 0 for (w, _) in self.ranks[w1][:k]]
#                print w1
#                print rs
#                print self.ranks[w1][:k]
#                print self.eval[w1]
                score += metrics.precision_at_k(rs, k)
            elif metric == 'map':
                rs = [1 if w in [x[0] for x in self.eval[w1]] else 0 for (w, _) in self.ranks[w1]]
#                print w1
#                print rs
#                print self.ranks[w1]
#                print self.eval[w1]
                score += metrics.mean_average_precision(rs)
            elif metric == 'ndcg_at_100':
                k = 100
                d = dict(self.eval[w1])
                rs = [d[w] if w in d else 0 for (w, s) in self.ranks[w1][:k]]
#                print w1
#                print rs
#                print self.ranks[w1][:k]
#                print self.eval[w1]
                score += metrics.ndcg_at_k(rs, k)

            n += 1

        return (score / n, n)
