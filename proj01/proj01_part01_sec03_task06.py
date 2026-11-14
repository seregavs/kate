import numpy as np
import gensim.downloader as api

# Загрузим эмбеддинги GloVe
glove_vectors = api.load("glove-wiki-gigaword-100")

print(f"Эмбеддинги GloVe для {len(glove_vectors)} слов\n")
print(f"Размерность эмбеддинга: {len(glove_vectors['hello'])}\n")
print(f"Пример: эмбеддинг для слова 'hello' {glove_vectors['hello']}\n")


def top5(target: np.ndarray, embeddings=glove_vectors) -> dict:
    # """
    #   Нахождение 5 ближайших слов для таргетного вектора.

    #   Args:
    #     target: произвольный вектор размера 100
    #     embeddings: эмбеддинги, которые используются для подсчета расстояний между словами

    #   Returns:
    #     список из 5 ближайших слов, упорядоченный по расстоянию от таргета
    #     (по возрастанию расстояния)

    #   Подсказка:
    #     используй функцию vector_distance, которую ты реализовал(а) ранее
    # """

    # напиши свое решение здесь

    # return top_words
    return 0


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