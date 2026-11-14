import numpy as np
import gensim.downloader as api
import string

def one_hot_embedding(text: str) -> dict:
#   """
#     Построение one-hot эмбеддингов для слов из заданного текста.

#     Args:
#       text: строка с латинскими буквами и знаками препинания

#     Returns:
#       словарь формата {слово: эмбеддинг}. Эмбеддинг должен иметь формат numpy array

#     """

    valid_symbols = 'abcdefghijklmnopqrstuvwxyz'

    # Приводим к нижнему регистру, заменяем остальные символы пробелами.
    cleaned_text = ''.join(
        char if (char in valid_symbols) else ' '
        for char in text.lower()
    )
    words = cleaned_text.split()
    words.sort()

    # Сохраняем уникальные слова в порядке первого появления.
    vocabulary = list(dict.fromkeys(words))
    vocabulary_size = len(vocabulary)
    # print(vocabulary)
    embeddings = {}

    for index, word in enumerate(vocabulary):
        embedding = np.zeros(vocabulary_size, dtype=int)
        embedding[index] = 1
        embeddings[word] = embedding

    return embeddings

# """
# Тесты для самопроверки. Изменять код в этой ячейке запрещено.
# Если ты правильно решил(а) задание, эта ячейка должна исполниться без ошибок
# и выдать вердикт «Тесты пройдены».
# """

text = "GPT is transforming education by providing instant access to knowledge and personalized learning. GPT is great!"

embeddings = one_hot_embedding(text)

flag = True

assert all(embeddings["access"] == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
assert all(embeddings["gpt"] == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
assert all(embeddings["transforming"] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
assert len(embeddings) == 14

print("Тесты пройдены корректно")