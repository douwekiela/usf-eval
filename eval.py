'''
Main evaluation utility stub
'''

import sys
import json

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python %s in_file' % sys.argv[0]
        quit()

    usf = json.load(open('usf-eval.json'))
