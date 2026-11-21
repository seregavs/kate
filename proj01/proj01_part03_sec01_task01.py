import numpy as np
import pandas as pd
import gensim.downloader as api
from operator import itemgetter

glove_vectors = api.load("glove-wiki-gigaword-100")

# Список слов, для которых существуют эмбеддинги GloVe
glove_keys = set(glove_vectors.key_to_index.keys())

def vector_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    squared_sum = 0.0

    for i in range(len(vec1)):
        squared_sum += (vec1[i] - vec2[i]) ** 2

    distance = float(np.sqrt(squared_sum))
    return distance

def scalar_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    res = 0.0
    for i in range(len(vec1)):
        res += vec1[i]*vec2[i]
    return res 


def top5(target: np.ndarray, embeddings=glove_vectors) -> dict:

    wd ={}
    wdl = []
    for word in embeddings.key_to_index:
        wd = {'word': word, 'distance': vector_distance(target, embeddings[word])}
        wdl.append(wd)

    wdl = sorted(wdl, key=itemgetter("distance"))
    res = [item['word'] for item in wdl[:6]]
    return res


prog = glove_vectors['programmer']
res1 = top5(prog)
print(res1)
m = glove_vectors['man']
w = glove_vectors['woman']
res2 = prog - m + w
res3 = top5(res2, glove_vectors)
print(res3)

d = m - w
lambda1 = scalar_product(prog, d) / scalar_product(d, d)
print(lambda1)