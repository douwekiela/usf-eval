'''
Main evaluation utility
'''

import sys, os
import json

from eval import model, metrics

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python %s in_file' % sys.argv[0])
        quit()

    in_file = sys.argv[1]
    if not os.path.exists(in_file):
        print('Input file does not exist')
        quit()

    usf = json.load(open('data/usf-eval.json'))

    me = model.ModelEval(usf, in_file)

    for metric in ['precision_at_10', 'map', 'ndcg_at_100']:
        score = me.evalMetric(metric)
        print metric, score
