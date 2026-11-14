import numpy as np
import gensim.downloader as api


def vector_cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
  # """
  # Вычисление косинуса угла между векторами. Считай, что
  # на вход поступают два ненулевых вектора одной длины.

  # Args:
  #   vec1: первый вектор
  #   vec2: второй вектор

  # Returns:
  #   косинус угла между векторами

  # Твоя функция может использовать метод np.sqrt. Никакие другие методы numpy
  # использовать нельзя.
  # """
    dot_product = 0.0
    norm1_squared = 0.0
    norm2_squared = 0.0

    for i in range(len(vec1)):
        x = float(vec1[i])
        y = float(vec2[i])

        dot_product += x * y
        norm1_squared += x * x
        norm2_squared += y * y

    cosine = dot_product / (
        np.sqrt(norm1_squared) * np.sqrt(norm2_squared)
    )

    return float(cosine)

# """
# Тесты для самопроверки. Изменять код в этой ячейке запрещено.
# Если ты правильно решил(а) задание, эта ячейка должна исполниться без ошибок
# и выдать вердикт «Тесты пройдены».
# """

import scipy

flag = True

for n in range(2, 10):

  v1 = np.random.normal(0, 1, n)
  v2 = np.random.normal(0, 1, n)

  if np.abs(vector_cosine(v1, v2) - (1 - scipy.spatial.distance.cosine(v1, v2))) > 0.001:
    print("Тест пройден неверно")
    print(f"v1: {v1}")
    print(f"v1: {v2}")
    flag = False
    break

if flag:
  print("Тесты пройдены корректно")