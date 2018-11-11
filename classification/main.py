from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import pymystem3
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns

from classification.experiments.onevsrest import OneVsRestExperiments

logging.basicConfig(filename='main.log', level=logging.INFO)
logging.warning('Start main!')
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset

from classification.experiments.grboost import GradientBoostingExperiments
from classification.experiments.knn import KNeighborsExperiments
from classification.experiments.logreg import LogisticRegressionExperiments
from classification.experiments.svc import SVCExperiments
from classification.experiments.voting import VotingExperiments
from classification.marking import mark_corpus, mark_corpus_multi_labels, clean_label, merge_marking
from classification.source import DataSource
from classification.tokenization import Tokenizer, TokensProvider
from classification.vectorization import Vectorizer, VectorsProvider
from classification.visualisation import Visualizer

pymystem3.mystem.MYSTEM_DIR = "/home/mluser/anaconda3/envs/master8_env/.local/bin"
pymystem3.mystem.MYSTEM_BIN = "/home/mluser/anaconda3/envs/master8_env/.local/bin/mystem"


def run_experiments(corpus_name, x_column_name, y_column_name):
    try:
        data_source = DataSource(corpus_name,
                                 x_column_name,
                                 y_column_name)

        # tokenizer = Tokenizer(data_source=data_source,
        #                       corpus_name=corpus_name)
        # tokenizer.tokenize()
        #
        # tokens_provider = TokensProvider(corpus_name=corpus_name)
        #
        # vectorizer = Vectorizer(tokens_provider=tokens_provider,
        #                         corpus_name=corpus_name)
        # vectorizer.vectorize_with_tfidf()
        # vectorizer.vectorize_with_w2v()

        vectors_provider = VectorsProvider(corpus_name=corpus_name)
        visualizer = Visualizer(corpus_name=corpus_name)

        onevsrest = OneVsRestExperiments(data_source=data_source,
                                         vectors_provider=vectors_provider,
                                         visualizer=visualizer)
        onevsrest.make_use_w2v()
        onevsrest.make_use_tfidf()
    except:
        logging.warning('FAILED - ' + corpus_name)


# n - name
# r - requirements
# d - duties
# all - all description + name

# sz - count vacancies per mark
# m - count marks
CURRENT_CORPUS_NAME = 'hh_corpus_sz245_m20_all_v5'

CURRENT_X_COLUMN_NAME = 'all_description'
CURRENT_Y_COLUMN_NAME = 'standard_mark'

# run_experiments(corpus_name='hh_corpus_sz245_m20_all',
#                 x_column_name='all_description',
#                 y_column_name='standard_mark')
#
# run_experiments(corpus_name='hh_corpus_sz245_m20_all_v2',
#                 x_column_name='all_description',
#                 y_column_name='standard_mark')
#
# run_experiments(corpus_name='hh_corpus_sz245_m20_all_v3',
#                 x_column_name='all_description',
#                 y_column_name='standard_mark')
#
data_source = DataSource(CURRENT_CORPUS_NAME,
                         CURRENT_X_COLUMN_NAME,
                         CURRENT_Y_COLUMN_NAME)

# tokenizer = Tokenizer(data_source=data_source,
#                       corpus_name=CURRENT_CORPUS_NAME)
# tokenizer.tokenize()
#
# tokens_provider = TokensProvider(corpus_name=CURRENT_CORPUS_NAME)
#
# vectorizer = Vectorizer(tokens_provider=tokens_provider,
#                         corpus_name=CURRENT_CORPUS_NAME)
# vectorizer.vectorize_with_tfidf()
# vectorizer.vectorize_with_w2v()
# vectorizer.vectorize_with_w2v_tfidf()
# vectorizer.vectorize_with_w2v_big()
# vectorizer.vectorize_with_tfidf_wshingles()
# vectorizer.vectorize_with_tfidf_ngrams()
# vectorizer.vectorize_with_w2v_old()

# vectors_provider = VectorsProvider(corpus_name=CURRENT_CORPUS_NAME)
# visualizer = Visualizer(corpus_name=CURRENT_CORPUS_NAME)
#
# onevsrest = OneVsRestExperiments(data_source=data_source,
#                                  vectors_provider=vectors_provider,
#                                  visualizer=visualizer)
# onevsrest.make_use_tfidf()
# onevsrest.make_use_w2v()

# x_all = vectors_provider.get_w2v_vectors()
# y_all = data_source.get_y_multi_label()
#
#
# model = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', n_jobs=-1), n_jobs=-1)
# logging.warning('Start w2v proba!')
# proba = cross_val_predict(model, x_all, y_all, cv=KFold(n_splits=5, shuffle=True), method='predict_proba', n_jobs=-1)
# logging.warning('end w2v proba!')
# proba = pd.DataFrame(proba)
# proba.to_csv('../data/new/predict_proba_w2v.csv')
# logging.warning('saved w2v proba!')




# x_all = vectors_provider.get_tfidf_vectors()
# y_all = data_source.get_y_multi_label()
#
#
# model = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', n_jobs=-1), n_jobs=-1)
# logging.warning('Start tfidf proba!')
# proba = cross_val_predict(model, x_all, y_all, cv=KFold(n_splits=5, shuffle=True), method='predict_proba', n_jobs=-1)
# logging.warning('end tfidf proba!')
# proba = pd.DataFrame(proba)
# proba.to_csv('../data/new/predict_proba_tfidf.csv')
# logging.warning('saved tfidf proba!')

# logreg = LogisticRegressionExperiments(data_source=data_source,
#                                        vectors_provider=vectors_provider,
#                                        visualizer=visualizer)
# logreg.make_use_w2v()
# logreg.make_use_tfidf()
# logreg.make_use_w2v_with_tfidf(
# logreg.make_use_w2v_big())
# logreg.make_use_tfidf_wshingles()
# logreg.make_use_tfidf_ngrams()
# logreg.make_use_w2v_old()

# run_experiments(corpus_name='hh_corpus_sz245_m20_all_v4',
#                 x_column_name='all_description',
#                 y_column_name='standard_mark')

# knn = KNeighborsExperiments(data_source=data_source,
#                             vectors_provider=vectors_provider,
#                             visualizer=visualizer)
# knn.make_use_tfidf()
# knn.make_use_w2v()
# knn.make_use_w2v_with_tfidf()
# knn.make_use_tfidf_wshingles()
# knn.make_use_tfidf_ngrams()
# knn.make_use_w2v_big()
# knn.make_use_w2v_old()

# svc = SVCExperiments(data_source=data_source,
#                      vectors_provider=vectors_provider,
#                      visualizer=visualizer)
# svc.make_use_tfidf()
# svc.make_use_w2v()
# svc.make_use_w2v_with_tfidf()
# svc.make_use_w2v_big()
# svc.make_use_tfidf_wshingles()
# svc.make_use_tfidf_ngrams()
# svc.make_use_w2v_old()

# voting = VotingExperiments(data_source=data_source,
#                            vectors_provider=vectors_provider,
#                            visualizer=visualizer)
# voting.make_use_tfidf()
# voting.make_use_w2v()
# voting.make_use_w2v_with_tfidf()
# voting.make_use_tfidf_wshingles()
# voting.make_use_tfidf_ngrams()
# voting.make_use_w2v_big()

# grboost = GradientBoostingExperiments(data_source=data_source,
#                                       vectors_provider=vectors_provider,
#                                       visualizer=visualizer)
# grboost.make_use_tfidf()
# grboost.make_use_w2v()
# grboost.make_use_w2v_with_tfidf()
# grboost.make_use_tfidf_wshingles()
# grboost.make_use_tfidf_ngrams()
# grboost.make_use_w2v_big()

# data = pd.read_csv("../data/new/hh_corpus_sz245_m20_all_v3.csv", header=0)
# data = mark_corpus(data)
# data.to_csv("../data/new/hh_all_corpus.csv", index=False, sep='|')
# data.to_csv("../data/new/hh_corpus_sz245_m20_all_confusion.csv", index=False, sep='|')
# def clean(str):
#     pattern = re.compile('<.*?>')
#     return pattern.sub('', str)
# size = 245
#
# data['name_requirements_duties'] = data.name + ' ' + data.requirements + ' ' + data.duties
# data['all_description'] = data.name + ' ' + data.description.apply(clean)
# data = data.drop_duplicates('all_description')
#
# frames = []
#
# for mark in range(1, 22):
# # for mark in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19, 21]:
#     if mark != 13:
#         frames.append(data[data.standard_mark == mark].sample(n=size))
#
# result = pd.concat(frames)
#
# result.to_csv("../data/new/hh_corpus_sz" + str(size) + "_m20_all.csv", index=False, sep='|')

# data_test = pd.read_csv("../data/new/marked_vacancies_hh_sz50_m16_nrd.csv", header=0, sep='|')
# result_test = data_test.groupby(['standard_mark'])[['name_requirements_duties']].describe()

# results = pd.read_csv('results/classification_results.csv', header=0)
# results = results[results.model_name != 'model1']
# df = results[results.dataset.str.contains('m16')].groupby(['vec_method', 'dataset']).max()
# # results[~results.dataset.str.contains('m16')].groupby(['vec_method', 'dataset']).cross_val_f1.max()
# df.reset_index(inplace=True)
# sns.barplot(x='vec_method', hue='dataset', y='cross_val_f1', palette="ch:.25", data=df)
# plt.legend(loc='lower left')
# plt.savefig('results/plots/dif_size.svg', format='svg')


# merge_marking(
#     corpus_original_name='hh_corpus_sz245_m20_all_v4.csv',
#     corpus_edited_name='hh_corpus_sz245_m20_all_v4_edit.csv',
#     corpus_result_name='hh_corpus_sz245_m20_all_v5.csv'
# )


# y = ad.apply(str.split, sep=',')


# co = mark_corpus_multi_labels(data)


# data[data.labels != ''].labels.value_counts()

# ly = data[data.labels != ''].labels.apply(str.split, sep=',')
# y = MultiLabelBinarizer().fit_transform(ly)


# temp = data[data.standard_mark != data.pred_mark_w2v].sort_values(by='proba_pred_w2v', ascending=False)


# model = BinaryRelevance(classifier=LogisticRegression(C=1.0, solver='sag', n_jobs=-1))
# model = LabelPowerset(classifier=LogisticRegression(C=1.0, solver='sag', n_jobs=-1)) # w2v 0.8342391573578064

def get_main(label: str):
    t = label.split(',')
    t = list(map(str.strip, t))

    return int(t[0])