import numpy as np
import pandas as pd
import gensim.downloader as api
from pathlib import Path
import nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import random

# gdown 1ZHGyWJnhLcXhEbhDo6ALK0tsa73yJITn

nltk.download('stopwords')
# Initialize the stopwords
stopset = set([x.lower() for x in stopwords.words('english')])
# Загрузим эмбеддинги GloVe
glove_vectors = api.load("glove-wiki-gigaword-100")
# Список слов, для которых существуют эмбеддинги GloVe
glove_keys = set(glove_vectors.key_to_index.keys())


def clean_news(text, sw = stopset):
    words = word_tokenize(text, preserve_line=True)
    return [word for word in words if word.lower() not in sw ]


def create_embedding(word_list):
  total = [0.0] * 100
  c = 0
  for w in word_list:
    if w in glove_keys:
      v = glove_vectors[w]
      for i in range(100):
        total[i] = total[i] + v[i]
      c += 1
  result = []
  for i in range(100):
    result.append(total[i]/c)
  return np.array(result)


def cluster_chk(df, cluster_field_name: str):
    category_codes, _ = pd.factorize(df["category"])
    cluster_codes, _ = pd.factorize(df[cluster_field_name])
    ari = adjusted_rand_score(category_codes, cluster_codes)
    ami = adjusted_mutual_info_score(category_codes, cluster_codes)
    print(f"Adjusted Rand Index 2:         {ari:.4f}")
    print(f"Adjusted Mutual Information 2: {ami:.4f}")

    # Inspect which categories appear in each cluster
    counts = pd.crosstab(df["category"], df[cluster_field_name])
    print("\nCategory × cluster counts:")
    print(counts)


def vector_distance(vec1: list, vec2: list) -> float:
    squared_sum = 0.0
    for i in range(len(vec1)):
        squared_sum += (vec1[i] - vec2[i]) ** 2
    distance = float(np.sqrt(squared_sum))
    return distance


def mykmeans(df, n_clusters: int = 3):
  centroids = df.sample(n=n_clusters, random_state=42).reset_index(drop=True)['news_embedding'].tolist()
  iteration = 0
  while True:
    change_count = 0
    iteration += 1
    for i in range(len(df)):
      row = df.iloc[i]
      dist = [0.0]*n_clusters
      for j in range(0, n_clusters):
        dist[j] = vector_distance(row["news_embedding"], centroids[j] )
      min_index = dist.index(min(dist))
      if iteration > 1:
        prev_cluster2 = df.loc[df.index[i], 'cluster2']
        if prev_cluster2 != min_index:
          change_count += 1
          df.loc[df.index[i], 'cluster2'] = min_index
      else:
        change_count += 1
        df.loc[df.index[i], 'cluster2'] = min_index

    print(f"iteration = {iteration}, change_count = {change_count}")
    if ( change_count == 0 ) or ( iteration > 99 ):
      print(df)
      return df

    for j in range(n_clusters):
        c = [0.0] * 100
        cluster_size = 0

        for i in range(len(df)):
            row = df.iloc[i]
            if row["cluster2"] == j:
                cluster_size += 1
                for k in range(100):
                    c[k] += row["news_embedding"][k]
        if cluster_size > 0:
            centroids[j] = [value / cluster_size for value in c]


fp = Path(__file__).resolve().parent / "content" / "bbc-news-data.csv"
df = pd.read_csv(fp, sep='\t')

#2. Оставим новости по трем категориям: 'sport', 'politics', 'entertainment'
df = df[df["category"].isin(['sport', 'politics', 'entertainment'])]

#3. Оставим для работы 500 случайных новостей

df = df.sample(n=500, random_state=42).reset_index(drop=True)
df["content"] = df["content"].apply(lambda x: re.sub(r"[^a-z]", " ", x.strip().lower()))
df["words"] = df["content"].apply(clean_news)
df["ID"] = df.index
df.head()
print(df)

# task 002 получим эмбеддинги для каждой новости
df["news_embedding"] = df["words"].apply(create_embedding)
print(df)

#Подготовим эмбеддинги новостей для кластеризации
news_embeddings = np.array(list(df["news_embedding"]))
#Кластеризуем новости на три кластера. Результаты запишем в поле cluster датасета
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(news_embeddings)
df['cluster'] = clusters
print(df)
cluster_chk(df, "cluster")
df = mykmeans(df, 3)
cluster_chk(df, "cluster2")
