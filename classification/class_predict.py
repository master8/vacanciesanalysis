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

import pickle
logging.basicConfig(filename='class_predict.log', level=logging.INFO)
logging.warning('Start predict!')
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

CURRENT_CORPUS_NAME = 'hh_all_corpus'

CURRENT_X_COLUMN_NAME = 'all_description'
CURRENT_Y_COLUMN_NAME = 'standard_mark'

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19',
                   '20', '21']

# data_source = DataSource(CURRENT_CORPUS_NAME,
#                              CURRENT_X_COLUMN_NAME,
#                              CURRENT_Y_COLUMN_NAME)
#
# vectors_provider = VectorsProvider(corpus_name=CURRENT_CORPUS_NAME)
#
# y_all = data_source.get_y_multi_label()
# x_all = vectors_provider.get_w2v_vectors_cbow()
#
# binarizer = MultiLabelBinarizer(classes=classes)
# y_all = binarizer.fit_transform(y_all)
# x_all = np.array(x_all)
#
# model = LabelPowerset(LogisticRegression(n_jobs=-1))
# model.fit(x_all, y_all)
#
# outfile = open('prepared_data/model.pkl', 'wb')
# pickle.dump(model, outfile)
# outfile.close()

data_source = DataSource(CURRENT_CORPUS_NAME,
                             CURRENT_X_COLUMN_NAME,
                             CURRENT_Y_COLUMN_NAME)

tokenizer = Tokenizer(data_source=data_source,
                      corpus_name=CURRENT_CORPUS_NAME)
tokenizer.tokenize()

tokens_provider = TokensProvider(corpus_name=CURRENT_CORPUS_NAME)

vectorizer = Vectorizer(tokens_provider=tokens_provider,
                        corpus_name=CURRENT_CORPUS_NAME)
vectorizer.vectorize_with_w2v_cbow()

vectors_provider = VectorsProvider(corpus_name=CURRENT_CORPUS_NAME)

X = np.array(vectors_provider.get_w2v_vectors_cbow())
corpus = data_source.get_corpus()

file = open('prepared_data/model.pkl', 'rb')
model = pickle.load(file)
file.close()

probabilities = model.predict_proba(X)
probabilities = pd.DataFrame(csr_matrix(probabilities).toarray(order=classes), columns=classes)

probabilities['predict_labels'] = ''

for mark in classes:
    probabilities.predict_labels = probabilities.predict_labels + probabilities[mark].apply(Evaluator.check_and_add_label, mark=mark)

probabilities.predict_labels = probabilities.predict_labels.apply(clean_label)
probabilities['vacancy_id'] = corpus.id

probabilities.to_csv('results/predict.csv', index=False)
