import numpy as np
import gensim.downloader as api


def vector_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
  # """
  # Вычисление расстояния между векторами. Считай, что
  # на вход поступают векторы одной длины.

  # Args:
  #   vec1: первый вектор
  #   vec2: второй вектор

  # Returns:
  #   расстояние между векторами

  # Твоя функция может использовать метод np.sqrt. Никакие другие методы numpy
  # использовать нельзя.
  # """

  # напиши свое решение здесь
    squared_sum = 0.0

    for i in range(len(vec1)):
        squared_sum += (vec1[i] - vec2[i]) ** 2

    distance = float(np.sqrt(squared_sum))
    return distance

# """
# Тесты для самопроверки. Изменять код в этой ячейке запрещено.
# Если ты правильно решил(а) задание, эта ячейка должна исполниться без ошибок
# и выдать вердикт «Тесты пройдены».
# """

flag = True

for n in range(1, 10):

  v1 = np.random.normal(0, 1, n)
  v2 = np.random.normal(0, 1, n)

  if np.abs(vector_distance(v1, v2) - np.linalg.norm(v1-v2)) > 0.001:
    print("Тест пройден неверно")
    print(f"v1: {v1}")
    print(f"v1: {v2}")
    flag = False
    break

if flag:
  print("Тесты пройдены корректно")