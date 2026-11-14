import numpy as np
import gensim.downloader as api


def vector_diff(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:

#   Вычисление разности двух векторов: vec1 - vec2. Считай, что
#   на вход поступают векторы одной длины.

#   Args:
#     vec1: первый вектор
#     vec2: второй вектор

#   Returns:
#     вектор разности (numpy array)

#   Твоя функция не должна использовать никаких встроенных методов numpy


  # напиши свое решение здесь
    diff = vec1 - vec2
    return diff

# """
# Тесты для самопроверки. Изменять код в этой ячейке запрещено.
# Если ты правильно решил(а) задание, эта ячейка должна исполниться без ошибок
# и выдать вердикт «Тесты пройдены».
# """
flag = True

for n in range(1, 10):

  v1 = np.random.normal(0, 1, n)
  v2 = np.random.normal(0, 1, n)

  if not np.allclose(vector_diff(v1, v2), v1-v2, 0, 0.001):
    print("Тест пройден неверно")
    print(f"v1: {v1}")
    print(f"v1: {v2}")
    flag = False
    break

if flag:
  print("Тесты пройдены корректно")