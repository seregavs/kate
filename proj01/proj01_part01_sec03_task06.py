import numpy as np
import gensim.downloader as api
from operator import itemgetter

# Загрузим эмбеддинги GloVe
glove_vectors = api.load("glove-wiki-gigaword-100")

print(f"Эмбеддинги GloVe для {len(glove_vectors)} слов\n")
print(f"Размерность эмбеддинга: {len(glove_vectors['hello'])}\n")
print(f"Пример: эмбеддинг для слова 'paris' {glove_vectors['paris']}\n")

def vector_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    squared_sum = 0.0

    for i in range(len(vec1)):
        squared_sum += (vec1[i] - vec2[i]) ** 2

    distance = float(np.sqrt(squared_sum))
    return distance


def top5(target: np.ndarray, embeddings=glove_vectors) -> dict:

    wd ={}
    wdl = []
    for word in embeddings.key_to_index:
        wd = {'word': word, 'distance': vector_distance(target, embeddings[word])}
        wdl.append(wd)

    wdl = sorted(wdl, key=itemgetter("distance"))
    res = [item['word'] for item in wdl[:5]]
    return res

# """
# Тесты для самопроверки. Изменять код в этой ячейке запрещено.
# Если ты правильно решил(а) задание, эта ячейка должна исполниться без ошибок
# и выдать вердикт «Тесты пройдены».
# """

target1 = glove_vectors['paris']
assert top5(target1, glove_vectors) == ['paris', 'london', 'france', 'brussels', 'rome']

target2 = glove_vectors['cat']
assert top5(target2, glove_vectors) == ['cat', 'dog', 'rabbit', 'cats', 'monkey']

target3 = glove_vectors['programmer']
assert top5(target3) == ['programmer', 'programmers', 'storyteller', 'technician', 'woodworker']

print("Тесты пройдены корректно")