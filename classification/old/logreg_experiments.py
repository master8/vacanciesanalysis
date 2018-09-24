from classification.visualisation.classification_results import show_classification_results

import numpy as np
import pandas
import pandas as pd
import gensim

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
import pymystem3

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

pymystem3.mystem.MYSTEM_DIR = "/home/mluser/anaconda3/envs/master8_env/.local/bin"
pymystem3.mystem.MYSTEM_BIN = "/home/mluser/anaconda3/envs/master8_env/.local/bin/mystem"
print(pymystem3.mystem.MYSTEM_DIR)
print(pymystem3.mystem.MYSTEM_BIN)
print(pymystem3.mystem.MYSTEM_EXE)

dataset_hh = pd.read_csv("../../data/old/old_marked_vacancies_from_hh.csv", header=0)
dataset_sj = pd.read_csv("../../data/old/old_marked_vacancies_from_sj.csv", header=0)




def TokenizeSentencesLemmatized(rawSentences):
    print('LEMMATIZED total = ' +str(rawSentences.__len__()))
    sentences = []
    m = Mystem()
    index = 0
    for c in rawSentences:
        tokenized_sents = m.lemmatize(c)
        cleaned_set = []
        for tokenized in tokenized_sents:
            if tokenized == "":
                break
            tokenized = tokenized.lower()
            if tokenized in stopwords.words('russian'):
                continue

            token = tokenized[0]
            if (token >= 'а' and token <= 'я'):
                cleaned_set.append(tokenized)
            elif ((token >= 'а' and token <= 'я') or (token >= 'a' and token <= 'z')):
                cleaned_set.append(tokenized)

        if cleaned_set.__len__()>0:
            sentences.append(cleaned_set)
        if index%1000 == 0:
          print(index)
        index+=1
    return sentences


hh_tokenized = TokenizeSentencesLemmatized(dataset_hh.requirements)
sj_tokenized = TokenizeSentencesLemmatized(dataset_sj.requirements)


def W2VStats(name, sentences):
    counter = [s.__len__() for s in sentences]
    print('Total n/o words in ' + name + ' : ' + str(sum(counter)))

W2VStats('merged', hh_tokenized + sj_tokenized)
w2v_hh_sj = gensim.models.Word2Vec(hh_tokenized + sj_tokenized, min_count=2, workers=2, iter=100, size=300, sg=0)


def SentenceToAverageWeightedVector(wv, sentence):
    vectors = pandas.DataFrame()
    index = 0
    try:
        for word in sentence:
            if word in wv.vocab:
                vectors[index] = wv[word]
            index += 1
        vectors = vectors.transpose()
        vector = vectors.mean().values.tolist()
    except Exception:
        return []
    return vector


vectors_w2v_hh_merged = [SentenceToAverageWeightedVector(w2v_hh_sj.wv, vacancy) for vacancy in hh_tokenized]
vectors_w2v_sj_merged = [SentenceToAverageWeightedVector(w2v_hh_sj.wv, vacancy) for vacancy in sj_tokenized]





x_all = vectors_w2v_sj_merged + vectors_w2v_hh_merged
y_all = pd.concat([dataset_sj.profession, dataset_hh.profession])


# w2c tf-idf 0.89
model1 = LogisticRegression(C=0.5, solver='liblinear')

# w2v
model2 = LogisticRegression(C=1.0, solver='sag')

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

model2.fit(x_train, y_train)
y_predict = model2.predict(x_test)

show_classification_results(y_test, y_predict)