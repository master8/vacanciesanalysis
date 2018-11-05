from datetime import datetime

import pandas as pd
import pymystem3
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
from classification.marking import mark_corpus, mark_corpus_multi_labels, clean_label
from classification.source import DataSource
from classification.tokenization import Tokenizer, TokensProvider
from classification.vectorization import Vectorizer, VectorsProvider
from classification.visualisation import Visualizer

pymystem3.mystem.MYSTEM_DIR = "/home/mluser/anaconda3/envs/master8_env/.local/bin"
pymystem3.mystem.MYSTEM_BIN = "/home/mluser/anaconda3/envs/master8_env/.local/bin/mystem"


def run_experiments(corpus_name, x_column_name, y_column_name):
    data_source = DataSource(corpus_name,
                             x_column_name,
                             y_column_name)

    tokenizer = Tokenizer(data_source=data_source,
                          corpus_name=corpus_name)
    tokenizer.tokenize()

    tokens_provider = TokensProvider(corpus_name=corpus_name)

    vectorizer = Vectorizer(tokens_provider=tokens_provider,
                            corpus_name=corpus_name)
    vectorizer.vectorize_with_tfidf()
    vectorizer.vectorize_with_w2v()
    vectorizer.vectorize_with_w2v_tfidf()
    vectorizer.vectorize_with_w2v_big()

    vectors_provider = VectorsProvider(corpus_name=corpus_name)
    visualizer = Visualizer(corpus_name=corpus_name)

    logreg = LogisticRegressionExperiments(data_source=data_source,
                                           vectors_provider=vectors_provider,
                                           visualizer=visualizer)
    logreg.make_use_tfidf()
    logreg.make_use_w2v()
    logreg.make_use_w2v_with_tfidf()
    logreg.make_use_w2v_big()

    svc = SVCExperiments(data_source=data_source,
                         vectors_provider=vectors_provider,
                         visualizer=visualizer)
    svc.make_use_tfidf()
    svc.make_use_w2v()
    svc.make_use_w2v_with_tfidf()
    svc.make_use_w2v_big()


# n - name
# r - requirements
# d - duties
# all - all description + name

# sz - count vacancies per mark
# m - count marks
CURRENT_CORPUS_NAME = 'hh_corpus_sz245_m20_all_v3'

CURRENT_X_COLUMN_NAME = 'all_description'
CURRENT_Y_COLUMN_NAME = 'standard_mark'
#
# run_experiments(corpus_name='hh_sz100_m20_nrd',
#                 x_column_name='name_requirements_duties',
#                 y_column_name='standard_mark')
#
# run_experiments(corpus_name='hh_sz100_m20_all',
#                 x_column_name='all_description',
#                 y_column_name='standard_mark')
#
#
# run_experiments(corpus_name='hh_sz50_m16_nrd',
#                 x_column_name='name_requirements_duties',
#                 y_column_name='standard_mark')
#
# run_experiments(corpus_name='hh_sz100_m16_nrd',
#                 x_column_name='name_requirements_duties',
#                 y_column_name='standard_mark')
#
# run_experiments(corpus_name='hh_sz200_m16_nrd',
#                 x_column_name='name_requirements_duties',
#                 y_column_name='standard_mark')
#
# run_experiments(corpus_name='hh_sz350_m16_nrd',
#                 x_column_name='name_requirements_duties',
#                 y_column_name='standard_mark')
#
# run_experiments(corpus_name='hh_sz500_m16_nrd',
#                 x_column_name='name_requirements_duties',
#                 y_column_name='standard_mark')

data_source = DataSource(CURRENT_CORPUS_NAME,
                         CURRENT_X_COLUMN_NAME,
                         CURRENT_Y_COLUMN_NAME)
#
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

vectors_provider = VectorsProvider(corpus_name=CURRENT_CORPUS_NAME)


# visualizer = Visualizer(corpus_name=CURRENT_CORPUS_NAME)

# x_all = vectors_provider.get_w2v_vectors()
# y_all = data_source.get_y()
#
# model1 = SVC(C=1, kernel='linear', probability=True)
# proba = cross_val_predict(model1, x_all, y_all, cv=KFold(n_splits=5, shuffle=True), method='predict_proba')
#
# method = 'w2v'
# proba = pd.DataFrame(proba)
# temp = pd.DataFrame()
#
# temp['true_mark'] = y_all
# temp['true_mark_index'] = temp.true_mark - 1
# temp.loc[temp.true_mark > 13, 'true_mark_index'] = temp.true_mark_index - 1
#
# proba_true = []
# for (i, row) in temp.iterrows():
#     proba_true.append(proba.iloc[i][int(row.true_mark_index)])
#
# proba_true_column = 'proba_true_' + method
# pred_mark_column = 'pred_mark_' + method
# proba_pred_column = 'proba_pred_' + method
#
# proba_true = pd.Series(proba_true)
# temp[proba_true_column] = proba_true
# temp = temp.drop(columns=['true_mark_index'])
# temp[pred_mark_column] = proba.idxmax(axis=1)
# temp[pred_mark_column] = temp.pred_mark_w2v + 1
# temp.loc[temp[pred_mark_column] > 12, pred_mark_column] = temp[pred_mark_column] + 1
# temp[proba_pred_column] = proba.max(axis=1)
#
# co = data_source.get_corpus()
# co['true_mark'] = temp['true_mark']
# co[proba_true_column] = temp[proba_true_column]
# co[pred_mark_column] = temp[pred_mark_column]
# co[proba_pred_column] = temp[proba_pred_column]
# data_source.save_corpus(co)

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


# data = pd.read_csv("../data/new/hh_corpus_sz245_m20_all_v2.csv", header=0, index_col='id')
# ed = pd.read_csv("../data/new/hh_corpus_sz245_m20_all_confusion_v2_edit.csv", header=0, index_col='id')
# co = pd.merge(data, ed, left_index=True, right_index=True, how='outer', suffixes=('', '_y'))
#
# co[co.main_label != 0].main_label.count()
# co[co.main_label_y != 0].main_label_y.count()
#
# co['main_label_y'] = co['main_label_y'].fillna(0).astype('int')
# co.loc[co.main_label_y != 0, 'main_label'] = co.main_label_y
#
# co['additional_labels_y'] = co['additional_labels_y'].fillna('0,0,0')
# co.loc[co.additional_labels_y != '0,0,0', 'additional_labels'] = co.additional_labels_y
#
# co['editor_name_y'] = co['editor_name_y'].fillna('')
# co.loc[co.editor_name_y != '', 'editor_name'] = co.editor_name_y
#
# co['comments_y'] = co['comments_y'].fillna('')
# co.loc[co.comments_y != '', 'comments'] = co.comments_y
#
# co = co[co.main_label != -1]
# co = co[co.main_label != 13]
#
# co.loc[co.main_label != 0, 'standard_mark'] = co.main_label
# co = co.drop(columns=['main_label_y', 'additional_labels_y', 'editor_name_y', 'comments_y'])
#
# co.to_csv("../data/new/hh_corpus_sz245_m20_all_v3.csv", index_label='id')


# y = ad.apply(str.split, sep=',')


# co = mark_corpus_multi_labels(data)


# data[data.labels != ''].labels.value_counts()

# ly = data[data.labels != ''].labels.apply(str.split, sep=',')
# y = MultiLabelBinarizer().fit_transform(ly)


# temp = data[data.standard_mark != data.pred_mark_w2v].sort_values(by='proba_pred_w2v', ascending=False)

# co.loc[co.additional_labels == '0,0,0', 'additional_labels'] = ''
# co.loc[co.main_label != 0, 'labels'] = co.main_label.apply(str) + ',' + co.additional_labels
#
# co.labels = co.labels.apply(clean_label)

binarizer = MultiLabelBinarizer()

x_all = vectors_provider.get_w2v_vectors()
y_all = data_source.get_y_multi_label()

y_all_bin = binarizer.fit_transform(y_all)

# model = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', n_jobs=-1), n_jobs=-1)
# model = BinaryRelevance(classifier=LogisticRegression(C=1.0, solver='sag', n_jobs=-1))
model = LabelPowerset(classifier=LogisticRegression(C=1.0, solver='sag', n_jobs=-1)) # w2v 0.8342391573578064
cross_val_f1 = cross_val_score(estimator=model, X=x_all, y=y_all_bin, scoring='f1_weighted', cv=5, n_jobs=-1)
