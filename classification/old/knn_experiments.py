import pickle

from classification.old.reading_data import read_tokenized_all_data
from classification.old.reading_data import read_all_data
from classification.visualisation.classification_results import show_classification_results

import numpy as np
import pandas
import pandas as pd
import gensim

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
import pymystem3
from keras.preprocessing.text import Tokenizer

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


dataset_hh, dataset_sj = read_all_data()
hh_tokenized, sj_tokenized = read_tokenized_all_data()

# into vectors

def SentenceToAverageTfIdfWeightedVector(wv, sentence, tfidf, dictionary):
    vectors = pandas.DataFrame()
    index = 0
    try:
        for word in sentence:
            if word not in dictionary.keys():
                tf_idf = 0
            else:
                word_index = dictionary[word]
                tf_idf = tfidf[word_index]
            if word in wv.vocab:
                vectors[index] = wv[word]*tf_idf
            index += 1
        vectors = vectors.transpose()
        vector = vectors.mean().values.tolist()
    except Exception:
        return []
    return vector

def GetW2VTFIDFVectors(w2v_model, tfidf_model, tfidf_dict, vacancies):
  index = 0
  vectors = []
  for vacancy in vacancies:
    vector = SentenceToAverageTfIdfWeightedVector(w2v_model.wv, vacancy,tfidf_model[index], tfidf_dict)
    vectors.append(vector)
    index+=1
  return vectors

def get_texts_to_matrix(texts, max_features=0):
    tokenizer = Tokenizer(split=" ", lower=True)
    if max_features != 0:
        tokenizer = Tokenizer(split=" ", lower=True, num_words=max_features,char_level='True')

    tokenizer.fit_on_texts(texts)
    matrix_tfidf = tokenizer.texts_to_matrix(texts=texts, mode='tfidf')
    print('Количество текстов:', matrix_tfidf.shape[0])
    print('Количество токенов:', matrix_tfidf.shape[1])
    return matrix_tfidf, tokenizer.word_index

tfidf, dictionary = get_texts_to_matrix(hh_tokenized)
tfidf_hh = {'tfidf': tfidf, 'dictionary':dictionary}
vectors_tfidf_hh = tfidf_hh['tfidf']

tfidf, dictionary = get_texts_to_matrix(hh_tokenized+sj_tokenized)
tfidf_hh_sj = {'tfidf': tfidf, 'dictionary':dictionary}

hh_len = vectors_tfidf_hh.shape[0]

w2v_hh_sj = gensim.models.Word2Vec(hh_tokenized+sj_tokenized, min_count=2, workers=2, iter=100,size=300,sg=0)

vectors_w2vtfidf_hh_merged = GetW2VTFIDFVectors(w2v_hh_sj, tfidf_hh_sj['tfidf'][:hh_len], tfidf_hh_sj['dictionary'], hh_tokenized)
vectors_w2vtfidf_sj_merged = GetW2VTFIDFVectors(w2v_hh_sj, tfidf_hh_sj['tfidf'][hh_len:], tfidf_hh_sj['dictionary'], sj_tokenized)


# classification

x_all = vectors_w2vtfidf_hh_merged + vectors_w2vtfidf_sj_merged
y_all = pd.concat([dataset_hh.profession, dataset_sj.profession])

# Testing accuracy: 0.8528784648187633
# Testing F1 score: 0.8546050200379395
model1 = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

model1.fit(x_train, y_train)
y_predict = model1.predict(x_test)

show_classification_results(y_test, y_predict, "w2v_tf-idf_knn_model1")