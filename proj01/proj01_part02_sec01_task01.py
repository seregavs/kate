import numpy as np
import pandas as pd
import gensim.downloader as api
from pathlib import Path
import nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
# Initialize the stopwords
stopset = set([x.lower() for x in stopwords.words('english')])

def clean_news(text, sw = stopset):
    words = word_tokenize(text, preserve_line=True)
    return [word for word in words if word.lower() not in sw ]

fp = Path(__file__).resolve().parent / "content" / "bbc-news-data.csv"
df = pd.read_csv(fp, sep='\t')

#2. Оставим новости по трем категориям: 'sport', 'politics', 'entertainment'
df = df[df["category"].isin(['sport', 'politics', 'entertainment'])]

#3. Оставим для работы 500 случайных новостей

df = df.sample(n=500, random_state=42).reset_index(drop=True)
df["content"] = df["content"].apply(lambda x: re.sub(r"[^a-z]", " ", x.strip().lower()))
df["words"] = df["content"].apply(clean_news)
df["content"] = df["content"].apply(lambda x: re.sub(r"[^a-z]", " ", x))
df["ID"] = df.index
df.head()
print(df)