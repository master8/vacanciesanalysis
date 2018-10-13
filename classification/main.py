from datetime import datetime

import pandas as pd
import pymystem3
import logging

# logging.basicConfig(filename='main.log', level=logging.INFO)
# logging.warning('Start main!')

from classification.experiments.grboost import GradientBoostingExperiments
from classification.experiments.knn import KNeighborsExperiments
from classification.experiments.logreg import LogisticRegressionExperiments
from classification.experiments.svc import SVCExperiments
from classification.experiments.voting import VotingExperiments
from classification.tokenization import Tokenizer, TokensProvider
from classification.vectorization import Vectorizer, VectorsProvider

pymystem3.mystem.MYSTEM_DIR = "/home/mluser/anaconda3/envs/master8_env/.local/bin"
pymystem3.mystem.MYSTEM_BIN = "/home/mluser/anaconda3/envs/master8_env/.local/bin/mystem"


# Tokenizer().tokenize()

# vectorizer = Vectorizer()
# vectorizer.vectorize_with_tfidf()
# vectorizer.vectorize_with_w2v()
# vectorizer.vectorize_with_w2v_tfidf()
# vectorizer.vectorize_with_tfidf_wshingles()
# vectorizer.vectorize_with_tfidf_ngrams()
# vectorizer.vectorize_with_w2v_big()
# vectorizer.vectorize_with_w2v_old()


# logreg = LogisticRegressionExperiments()
# logreg.make_use_tfidf()
# logreg.make_use_w2v()
# logreg.make_use_w2v_with_tfidf()
# logreg.make_use_tfidf_wshingles()
# logreg.make_use_tfidf_ngrams()
# logreg.make_use_w2v_big()
# logreg.make_use_w2v_old()

# knn = KNeighborsExperiments()
# knn.make_use_tfidf()
# knn.make_use_w2v()
# knn.make_use_w2v_with_tfidf()
# knn.make_use_tfidf_wshingles()
# knn.make_use_tfidf_ngrams()
# knn.make_use_w2v_big()
# knn.make_use_w2v_old()

# svc = SVCExperiments()
# svc.make_use_tfidf()
# svc.make_use_w2v()
# svc.make_use_w2v_with_tfidf()
# svc.make_use_tfidf_wshingles()
# svc.make_use_tfidf_ngrams()
# svc.make_use_w2v_big()
# svc.make_use_w2v_old()

# voting = VotingExperiments()
# voting.make_use_tfidf()
# voting.make_use_w2v()
# voting.make_use_w2v_with_tfidf()
# voting.make_use_tfidf_wshingles()
# voting.make_use_tfidf_ngrams()
# voting.make_use_w2v_big()

# grboost = GradientBoostingExperiments()
# grboost.make_use_tfidf()
# grboost.make_use_w2v()
# grboost.make_use_w2v_with_tfidf()
# grboost.make_use_tfidf_wshingles()
# grboost.make_use_tfidf_ngrams()
# grboost.make_use_w2v_big()

data = pd.read_csv("../data/new/vacancies_hh_all_051018.csv", header=0, sep='|')


# data.loc[data.id == 20000000, 'custom_mark'] = 1
# data[data.name.str.contains('QA')].name[:50]
# data[data.specializations.str.contains('1.137')].name[:50]
