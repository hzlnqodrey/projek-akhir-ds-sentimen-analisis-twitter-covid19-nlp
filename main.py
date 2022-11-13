# !pip install Sastrawi
# !pip install tensorflow
# !pip install word2vec
# !pip install keras
# !pip install symspellpy

import re  # regex library
from symspellpy import SymSpell, Verbosity
from symspellpy import SymSpell
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import word2vec
import gzip
from collections import Counter
from urllib.request import urlopen
from gensim.models.word2vec import Word2Vec
from gensim.corpora import WikiCorpus
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
import keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping
import re
from keras.layers import Dense, Embedding, LSTM, Input, GRU, Bidirectional, Dropout
from keras.models import Model
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import array
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import ast
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
from wordcloud import WordCloud
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('punkt')

# %matplotlib inline

# Case Folding
df = pd.DataFrame()
df['komentar'] = df['komentar'].str.lower()
print('Case Folding Result : \n')
print(df['komentar'].head(5))
print('\n\n\n')

# Case Folding
df = pd.DataFrame()
df['komentar'] = ''
df['komentar'] = df['komentar'].str.lower()
print('Case Folding Result : \n')
print(df['komentar'].head(5))
print('\n\n\n')

# Tokenizing


def remove_comments_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t', " ").replace('\\n', " ").replace(
        '\\u', " ").replace('\\', " ").replace('.', " ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", "")


df['komentar'] = df['komentar'].apply(remove_comments_special)


def remove_number(text):
    return re.sub(r"\d+", " ", text)


df['komentar'] = df['komentar'].apply(remove_number)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


df['komentar'] = df['komentar'].apply(remove_punctuation)


def remove_whitespace_LT(text):
    return text.strip()


df['komentar'] = df['komentar'].apply(remove_whitespace_LT)


def remove_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)


df['komentar'] = df['komentar'].apply(remove_whitespace_multiple)


def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)


def word_tokenize_wrapper(text):
    return word_tokenize(text)


df['comments_tokens'] = df['komentar'].apply(word_tokenize_wrapper)
print('Tokenizing Result : \n')
print(df['comments_tokens'].head())
print('\n\n\n')

# Filtering
normalizad_word = pd.read_csv(
    "https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/kamus_singkatan.csv", sep=";", header=None)
normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]


def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


df['comments_normalized'] = df['comments_tokens'].apply(normalized_term)

normalizad_word2 = pd.read_csv(
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRQS3tlUL5EcxYqbbYzFLHmHaqm2npjYDLyz0dzwMIcUVhfoVWKuhR52P9YCqbAyY9zCgT66JVutWA/pub?output=csv", header=None)

normalizad_word_dict2 = {}
for index, row in normalizad_word2.iterrows():
    if row[0] not in normalizad_word_dict2:
        normalizad_word_dict2[row[0]] = row[1]


def normalized_term2(document):
    return [normalizad_word_dict2[term] if term in normalizad_word_dict2 else term for term in document]


df['comments_normalized'] = df['comments_normalized'].apply(normalized_term2)

list_stopwords = (['yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'seperti', 'jika', 'jika', 'sehingga', 'mungkin', 'kembali', 'dan', 'ini', 'karena', 'oleh', 'saat', 'sekitar', 'bagi', 'serta', 'di', 'dari', 'sebagai', 'hal', 'ketika', 'adalah', 'itu', 'dalam', 'bahwa', 'atau', 'dengan', 'akan', 'juga', 'kalau', 'ada', 'terhadap', 'secara', 'agar', 'lain', 'jadi', 'yang ', 'sudah', 'sudah begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi', 'maka', 'tentang', 'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya', 'guna', 'kah', 'pun', 'sampai', 'sedangkan', 'selagi',
                  'sementara', 'tetapi', 'apakah', 'sebab', 'selain', 'seolah', 'seraya', 'seterusnya', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu', 'demikian', 'tapi', 'juga', 'mari', 'nanti', 'melainkan', 'oh', 'ok', 'sebetulnya', 'setiap', 'sesuatu', 'pasti', 'saja', 'toh', 'ya', 'walau', 'apalagi', 'bagaimanapun', 'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'krn', 'nya', 'nih', 'sih', 'ah', 'ssh', 'om', 'ah', 'si', 'tau', 'tuh', 'utk', 'ya', 'cek', 'jd', 'aja', 't', 'nyg', 'hehe', 'pen', 'nan', 'loh', 'rt', '&amp', 'yah', 'ni', 'ret', 'za', 'nak', 'haa', 'zaa', 'maa', 'lg', 'eh', 'hmm', 'kali'])

list_stopwords = set(list_stopwords)


def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


df['comments_tokens_sw'] = df['comments_normalized'].apply(stopwords_removal)

# Spell Checker


def join_text_list(texts):
    texts = ast.literal_eval(texts)
    return ' '.join([text for text in texts])


df["comments_tokens_sw"] = df["comments_tokens_sw"].apply(join_text_list)


sym_spell = SymSpell()
corpus_path = "/content/drive/MyDrive/Final IFest Ngenkonst/wiki-id-formatted.txt"
sym_spell.create_dictionary(corpus_path)


input_term = "maksuddnya siaapa kam adlh seekr gjha"
suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=2)
for suggestion in suggestions:
    print(suggestion.term)

Comments = []

for index, row in df.iterrows():
    suggestions = sym_spell.lookup_compound(
        row["comments_tokens_sw"], max_edit_distance=2)
    Comments.append(suggestions[0].term)

df["Comments"] = Comments
df['Comments'] = df['Comments'].apply(word_tokenize_wrapper)
df.head()

# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def stemmed_wrapper(term):
    return stemmer.stem(term)


term_dict = {}

for document in df['Comments']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term, ":", term_dict[term])
print(term_dict)
print("------------------------")


def get_stemmed_term(document):
    return [term_dict[term] for term in document]


df['comments_tokens_stemmed'] = df['Comments'].apply(get_stemmed_term)

print(df['comments_tokens_stemmed'])

df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX1vTbpuC4sfbTZWhW0QFK0mjjUV7wzthziWDMCM91ifTXQc5emKHFJVIcgM4wI2cQULwmq0y2Crxnb1ee/pub?gid=850839976&single=true&output=csv', sep=',')

df
df['Content'] = df['Content_Clean']
id_w2v_2 = Word2Vec.load(
    '/content/drive/MyDrive/Dimas Ananda, S.Stat/SKRIPSI/idwiki_word2vec_100.model')

print('Vocab length:', len(w2v_model.wv.key_to_index))
print('Vector size:', (w2v_model.wv.vector_size))

EXP_EMBED_DIM = w2v_model.wv.vector_size
df.Content = df.Content.astype(str)

texts = df.Content
tokenizer = Tokenizer(oov_token=('unknown'))
tokenizer.fit_on_texts(texts)
sequences_train = tokenizer.texts_to_sequences(df.Content)
sample_oov_token = 'aneh'
print(sample_oov_token in tokenizer.word_index)
print(tokenizer.word_index['unknown'])
tokenizer.texts_to_sequences([sample_oov_token])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
EXP_VOCAB_SIZE = len(word_index)+1
EXP_MAX_FEATURES = EXP_VOCAB_SIZE

print('Word Index List :\n', word_index)

# Print sample of tokenized text
try:
    for i, tokenized in texts.sample(5).items():
        print(i, tokenized)
        print(sequences_train[i])
        print('\n')

    for i, tokenized in texts.sample(5).items():
        print(i, tokenized)
        print(sequences_test[i])
        print('\n')
except:
    print('-')

# Print sample of tokenized text
sample_tokenized = texts.sample(5)
for i, tokenized in sample_tokenized.items():
    print(tokenized)
    print(sequences_train[i])
    print('\n')
len(sequences_train)
embedding_matrix = np.zeros((EXP_VOCAB_SIZE, EXP_EMBED_DIM))

OOV = []  # out of vocabulary word
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    else:
        OOV.append(word)

print('NUMBER OF OOV:', len(OOV), OOV)
print('SHAPE OF EMBEDDING MATRIX :', embedding_matrix.shape)
word_index['covid']

print(f"Embedding matrix for word '{list(word_index.keys())[42]}': \n")

embedding_matrix[42]

# Splitting words in every tweets (similar with tokenizing)
Bigger_list = []
for i in df.Content:
    li = list(i.split(" "))
    Bigger_list.append(li)
print(Bigger_list[0:2])


def text_to_int(df, word_index, max_len):
    X = np.zeros((df.shape[0], max_len))  # initialising the nd-array
    for i, tweet in enumerate(df):
        words = list(tweet.split(" "))
        j = 0
        for word in reversed(words):  # reversed -> right aligned
            if word in word_index.keys():  # if present in our vocab
                X[i, max_len-1-j] = word_index[word]
                j += 1
    return X


# finding the longest word of tweet
max_len = 0
for list_ in Bigger_list:
    if len(list_) > max_len:
        max_len = len(list_)

# cari rata rata
print('Length of longest tweet is', max_len, 'words')

# converting train_data tweets to integer from word_index
X = text_to_int(df.Content, word_index, max_len)
print(X.shape)
print(df.Content[7], '\n mapped to \n', X[60])
print(df.Content[20], '\n mapped to \n', X[60])
df.shape[0]
