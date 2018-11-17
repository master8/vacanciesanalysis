from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, \
    PassiveAggressiveClassifier, LogisticRegressionCV, RidgeClassifierCV
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN, MLARAM
from skmultilearn.ensemble import RakelD, RakelO
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

from classification.evaluation import Evaluator
from classification.source import DataSource
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer
import logging
from datetime import datetime


class LabelPowersetExperiments:

    __CLASSIFIER_NAME = "LabelPowerset"

    def __init__(self,
                 data_source: DataSource,
                 vectors_provider: VectorsProvider,
                 visualizer: Visualizer):
        self.__data_source = data_source
        self.__vectors_provider = vectors_provider
        self.__visualizer = visualizer

    def make_use_tfidf(self):
        x_all = self.__vectors_provider.get_tfidf_vectors()
        y_all = self.__data_source.get_y_multi_label()

        # TODO here grid search

        base_estimators = [
            LogisticRegression(C=1.0, solver='sag', n_jobs=-1),
            LogisticRegression(n_jobs=-1),
            LinearSVC(),
            MLPClassifier()
        ]

        model_params = [
            "LogisticRegression(C=1.0, solver='sag')",
            "LogisticRegression()",
            "LinearSVC()",
            "MLPClassifier()"
        ]

        i = 0
        for base_estimator in base_estimators:
            logging.warning(str(datetime.now()) + 'Start ' + model_params[i])
            try:
                model = LabelPowerset(base_estimator)
                cross_val_f1 = Evaluator.evaluate_only_cross_val(model, x_all, y_all)
                self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, model_params[i],
                                                       "tfidf", cross_val_f1)
            except:
                logging.warning('Error on ' + model_params[i])
            logging.warning(str(datetime.now()) + 'End ' + model_params[i])
            i += 1

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__data_source.get_y_multi_label()

        # TODO here grid search

        base_estimators = [
            LogisticRegression(C=1.0, solver='sag', n_jobs=-1),
            LogisticRegression(n_jobs=-1),
            LinearSVC(),
            MLPClassifier()
        ]

        model_params = [
            "LogisticRegression(C=1.0, solver='sag')",
            "LogisticRegression()",
            "LinearSVC()",
            "MLPClassifier()"
        ]

        i = 0
        for base_estimator in base_estimators:
            logging.warning(str(datetime.now()) + 'Start ' + model_params[i])
            try:
                model = LabelPowerset(base_estimator)
                cross_val_f1 = Evaluator.evaluate_only_cross_val(model, x_all, y_all)
                self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, model_params[i],
                                                       "Word2Vec", cross_val_f1)
            except:
                logging.warning('Error on ' + model_params[i])
            logging.warning(str(datetime.now()) + 'End ' + model_params[i])
            i += 1

    def make_use_tfidf_with_results(self):
        x_all = self.__vectors_provider.get_tfidf_vectors()
        y_all = self.__data_source.get_y_multi_label()

        model1 = LabelPowerset(LogisticRegression(C=1.0, solver='sag', n_jobs=-1))
        Evaluator.multi_label_predict_proba_tfidf(model1, x_all, y_all, data_source=self.__data_source)

    def make_use_w2v_with_results(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__data_source.get_y_multi_label()

        model1 = LabelPowerset(LogisticRegression(C=1.0, solver='sag', n_jobs=-1))
        Evaluator.multi_label_predict_proba_w2v(model1, x_all, y_all, data_source=self.__data_source)
