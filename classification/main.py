from datetime import datetime

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import clone, cluster
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import pymystem3
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
from skmultilearn.ensemble import LabelSpacePartitioningClassifier

from classification.evaluation import Evaluator
from classification.experiments.labelpowerset import LabelPowersetExperiments
from classification.experiments.onevsrest import OneVsRestExperiments
from sklearn.metrics import classification_report

from classification.preprocessing import process_text
from skmultilearn.cluster import MatrixLabelSpaceClusterer
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, Birch
from sklearn.cluster import DBSCAN

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

from skmultilearn.cluster import LabelCooccurrenceGraphBuilder

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
CURRENT_CORPUS_NAME = 'hh_corpus_sz245_m20_all_v7_t'

CURRENT_X_COLUMN_NAME = 'all_description'
CURRENT_Y_COLUMN_NAME = 'standard_mark'


def run_size_experiments(n_samples):
    logging.warning('start' + str(n_samples))
    data_source = DataSource(CURRENT_CORPUS_NAME,
                             CURRENT_X_COLUMN_NAME,
                             CURRENT_Y_COLUMN_NAME, n_samples)

    # tokenizer = Tokenizer(data_source=data_source,
    #                       corpus_name=CURRENT_CORPUS_NAME)
    # tokenizer.tokenize()
    #
    # tokens_provider = TokensProvider(corpus_name=CURRENT_CORPUS_NAME)
    #
    # vectorizer = Vectorizer(tokens_provider=tokens_provider,
    #                         corpus_name=CURRENT_CORPUS_NAME)
    # vectorizer.vectorize_with_w2v_cbow()
    # vectorizer.vectorize_with_tfidf()

    vectors_provider = VectorsProvider(corpus_name=CURRENT_CORPUS_NAME)
    visualizer = Visualizer(corpus_name=CURRENT_CORPUS_NAME + '_' + str(n_samples))

    ovr = OneVsRestExperiments(data_source=data_source,
                               vectors_provider=vectors_provider,
                               visualizer=visualizer)
    ovr.make_use_d2v()
    # ovr.make_use_w2v()
    # ovr.make_use_tfidf()

    lpe = LabelPowersetExperiments(data_source=data_source,
                                   vectors_provider=vectors_provider,
                                   visualizer=visualizer)
    lpe.make_use_d2v_dbow()
    # lpe.make_use_w2v_cbow()
    # lpe.make_use_tfidf()
    logging.warning('end! ' + str(n_samples))


# run_size_experiments(None)
# run_size_experiments(10)
# run_size_experiments(20)
# run_size_experiments(30)
# run_size_experiments(40)
# run_size_experiments(50)
# run_size_experiments(60)
# run_size_experiments(70)
# run_size_experiments(80)
# run_size_experiments(90)
# run_size_experiments(100)
# run_size_experiments(110)
# run_size_experiments(120)
# run_size_experiments(130)
# run_size_experiments(140)
# run_size_experiments(150)
# run_size_experiments(160)
# run_size_experiments(170)
# run_size_experiments(180)
# run_size_experiments(190)
# run_size_experiments(200)

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
# data_source = DataSource(CURRENT_CORPUS_NAME,
#                          CURRENT_X_COLUMN_NAME,
#                          CURRENT_Y_COLUMN_NAME)

# tokenizer = Tokenizer(data_source=data_source,
#                       corpus_name=CURRENT_CORPUS_NAME)
# tokenizer.tokenize_simple()

# tokens_provider = TokensProvider(corpus_name=CURRENT_CORPUS_NAME)
#
# vectorizer = Vectorizer(tokens_provider=tokens_provider,
#                         corpus_name=CURRENT_CORPUS_NAME)
# vectorizer.vectorize_with_d2v_dbow()
# vectorizer.vectorize_with_w2v()
# vectorizer.vectorize_with_w2v_cbow()
# vectorizer.vectorize_with_w2v_fix()
# vectorizer.vectorize_with_tfidf()
# vectorizer.vectorize_with_w2v_tfidf()
# vectorizer.vectorize_with_w2v_big()
# vectorizer.vectorize_with_tfidf_wshingles()
# vectorizer.vectorize_with_tfidf_ngrams()
# vectorizer.vectorize_with_w2v_old()

# vectors_provider = VectorsProvider(corpus_name=CURRENT_CORPUS_NAME)
# visualizer = Visualizer(corpus_name=CURRENT_CORPUS_NAME)

# onevsrest = OneVsRestExperiments(data_source=data_source,
#                                  vectors_provider=vectors_provider,
#                                  visualizer=visualizer)
# onevsrest.make_use_w2v()
# onevsrest.make_use_tfidf()
#
# lpe = LabelPowersetExperiments(data_source=data_source,
#                                vectors_provider=vectors_provider,
#                                visualizer=visualizer)
# lpe.make_use_w2v_with_results()
# lpe.make_use_tfidf_with_results()
# lpe.make_use_w2v()
# lpe.make_use_w2v_cbow()
# lpe.make_use_w2v_fix()
# lpe.make_use_tfidf()


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

re = pd.read_csv('results/classification_reports_t.csv', header=0)
x = range(8, 161, 8)
data = pd.DataFrame(index=x)
data['LogisticRegression Word2Vec'] = np.array(re[(re.vec_method == 'Word2Vec_CBOW') & (re.model_params == 'LogisticRegression()')].f1_macro)
data['LogisticRegression TF-IDF'] = np.array(re[(re.vec_method == 'tfidf') & (re.model_params == 'LogisticRegression()')].f1_macro)
data['MLPClassifier Word2Vec'] = np.array(re[(re.vec_method == 'Word2Vec_CBOW') & (re.model_params == 'MLPClassifier()')].f1_macro)
data['LinearSVC TF-IDF'] = np.array(re[(re.vec_method == 'tfidf') & (re.model_params == 'LinearSVC()')].f1_macro)

cnn = [0.01666666666666667,
 0.45550793650793653,
 0.80284165181224,
 0.7415938911492613,
 0.8078647320342457,
 0.8237790749416138,
 0.8504831107679424,
 0.837000439516449,
 0.8442408376616315,
 0.8852001267198334,
 0.8478267928128046,
 0.8568798400644813,
 0.8555067068337143,
 0.8688222721392332,
 0.8746513361193348,
 0.8628899991793659,
 0.8835756003065001,
 0.8791513625861697,
 0.8736762527922641,
 0.8764629081069939]

#data['CNN'] = np.array(cnn)

sns.lineplot(data=data)
plt.savefig('results/plots/dif_size.png', format='png', dpi=300)


# merge_marking(
#     corpus_original_name='hh_corpus_sz245_m20_all_v5.csv',
#     corpus_edited_name='hh_corpus_sz245_m20_all_v5_edit.csv',
#     corpus_result_name='hh_corpus_sz245_m20_all_v7.csv'
# )


# y = ad.apply(str.split, sep=',')


# co = mark_corpus_multi_labels(data)


# data[data.labels != ''].labels.value_counts()

# ly = data[data.labels != ''].labels.apply(str.split, sep=',')
# y = MultiLabelBinarizer().fit_transform(ly)


# temp = data[data.standard_mark != data.pred_mark_w2v].sort_values(by='proba_pred_w2v', ascending=False)


# model = BinaryRelevance(classifier=LogisticRegression(C=1.0, solver='sag', n_jobs=-1))
# model = LabelPowerset(classifier=LogisticRegression(C=1.0, solver='sag', n_jobs=-1)) # w2v 0.8342391573578064

# def get_main(label: str):
#     t = label.split(',')
#     t = list(map(str.strip, t))
#
#     return int(t[0])

# x_all = vectors_provider.get_w2v_vectors()
# y_all = data_source.get_y_multi_label()
# model = LabelPowerset(LogisticRegression(n_jobs=-1))
# model = OneVsRestClassifier(LogisticRegression(n_jobs=-1), n_jobs=-1)
# classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19',
#                    '20', '21']
# binarizer = MultiLabelBinarizer(classes=classes)
# y_all = binarizer.fit_transform(y_all)
# x_all = np.array(x_all)
#
# x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred, target_names=classes))

# Evaluator.multi_label_report(model, x_all, y_all, sparse=True)

# tokenized_requirements = data_source.get_x()[:10].apply(lambda text: process_text(text)['lemmatized_text_pos_tags'])

# x_all = vectors_provider.get_w2v_vectors_cbow()
# y_all = data_source.get_y_multi_label()
# model = LabelPowerset(LogisticRegression(n_jobs=-1))
#
# classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19',
#            '20', '21']
# binarizer = MultiLabelBinarizer(classes=classes)
# y_all = binarizer.fit_transform(y_all)
# x_all = np.array(x_all)
#
# x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)
#
# graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
# edge_map = graph_builder.transform(y_train)
#
# # matrix_clusterer = MatrixLabelSpaceClusterer(clusterer=KMeans(n_clusters=20, n_jobs=-1))
# matrix_clusterer = MatrixLabelSpaceClusterer(clusterer=DBSCAN(eps=.2))
#
# matrix_clusterer.fit_predict(x_train, y_train)
#
# classifier = LabelSpacePartitioningClassifier(
#     classifier=model,
#     clusterer=matrix_clusterer
# )
#
# classifier.fit(x_train, y_train)
# prediction = classifier.predict(x_test)
# y_pred = csr_matrix(prediction).toarray(order=classes)
# print(classification_report(y_test, y_pred, target_names=classes))


# re = []
# for mark in classes:
#     co['has_' + mark] = co.labels.apply(lambda le: mark in le.split(','))
#     re.append(co[co['has_' + mark]].sample(10))
#     print(mark + ' ' + str(co[co['has_' + mark]].labels.count()))
#
# re = pd.concat(re)
