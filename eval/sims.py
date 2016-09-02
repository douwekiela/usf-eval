'''
Similarity functions
'''
import numpy as np

def norm(v):
    return np.linalg.norm(v)

def cosine(u, v):
    return np.dot(u, v) / (norm(u) * norm(v))
