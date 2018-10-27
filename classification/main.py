from datetime import datetime

import pandas as pd
import pymystem3
import logging
import re

# logging.basicConfig(filename='main.log', level=logging.INFO)
# logging.warning('Start main!')

from classification.experiments.grboost import GradientBoostingExperiments
from classification.experiments.knn import KNeighborsExperiments
from classification.experiments.logreg import LogisticRegressionExperiments
from classification.experiments.svc import SVCExperiments
from classification.experiments.voting import VotingExperiments
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
# CURRENT_CORPUS_NAME = 'hh_sz100_m20_rd'
#
# CURRENT_X_COLUMN_NAME = 'requirements_duties'
# CURRENT_Y_COLUMN_NAME = 'standard_mark'
#
run_experiments(corpus_name='hh_sz100_m20_nrd',
                x_column_name='name_requirements_duties',
                y_column_name='standard_mark')

run_experiments(corpus_name='hh_sz100_m20_all',
                x_column_name='all_description',
                y_column_name='standard_mark')


run_experiments(corpus_name='hh_sz50_m16_nrd',
                x_column_name='name_requirements_duties',
                y_column_name='standard_mark')

run_experiments(corpus_name='hh_sz100_m16_nrd',
                x_column_name='name_requirements_duties',
                y_column_name='standard_mark')

run_experiments(corpus_name='hh_sz200_m16_nrd',
                x_column_name='name_requirements_duties',
                y_column_name='standard_mark')

run_experiments(corpus_name='hh_sz350_m16_nrd',
                x_column_name='name_requirements_duties',
                y_column_name='standard_mark')

run_experiments(corpus_name='hh_sz500_m16_nrd',
                x_column_name='name_requirements_duties',
                y_column_name='standard_mark')

# data_source = DataSource(CURRENT_CORPUS_NAME,
#                          CURRENT_X_COLUMN_NAME,
#                          CURRENT_Y_COLUMN_NAME)

# tokenizer = Tokenizer(data_source=data_source,
#                       corpus_name=CURRENT_CORPUS_NAME)
# tokenizer.tokenize()

# tokens_provider = TokensProvider(corpus_name=CURRENT_CORPUS_NAME)

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

# logreg = LogisticRegressionExperiments(data_source=data_source,
#                                        vectors_provider=vectors_provider,
#                                        visualizer=visualizer)
# logreg.make_use_tfidf()
# logreg.make_use_w2v()
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

# data = pd.read_csv("../data/new/marked_vacancies_hh_all_131018.csv", header=0, sep='|')

# def clean(str):
#     pattern = re.compile('<.*?>')
#     return pattern.sub('', str)
# size = 500
#
# data['name_requirements_duties'] = data.name + ' ' + data.requirements + ' ' + data.duties
# data['all_description'] = data.name + ' ' + data.description.apply(clean)
# data = data.drop_duplicates('name_requirements_duties')
#
# frames = []

# for mark in range(1, 22):
# for mark in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19, 21]:
#     if mark != 13:
#         frames.append(data[data.standard_mark == mark].sample(n=size))
#
# result = pd.concat(frames)
#
# result.to_csv("../data/new/marked_vacancies_hh_sz" + str(size) + "_m16_nrd.csv", index=False, sep='|')

# data_test = pd.read_csv("../data/new/marked_vacancies_hh_sz50_m16_nrd.csv", header=0, sep='|')
# result_test = data_test.groupby(['standard_mark'])[['name_requirements_duties']].describe()
