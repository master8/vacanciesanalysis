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

from classification.evaluation import Evaluator
from classification.source import DataSource
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer
import logging
from datetime import datetime


class OneVsRestExperiments:

    __CLASSIFIER_NAME = "OneVsRestClassifier"

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
            # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
            #                    intercept_scaling=True, max_iter=100, multi_class='ovr', penalty='l2', random_state=None,
            #                    solver='newton-cg',
            #                    tol=0.0001, warm_start=False, n_jobs=-1),
            LinearSVC(),
            MLPClassifier(),
            # SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
            #     decision_function_shape='ovr', degree=0, gamma='auto', kernel='linear',
            #     max_iter=-1, probability=True, random_state=None, shrinking=True,
            #     tol=0.001, verbose=False)
        ]

        model_params = [
            "LogisticRegression(C=1.0, solver='sag')",
            'LogisticRegression()',
            # "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\
            #                    intercept_scaling=True, max_iter=100, multi_class='ovr', penalty='l2', random_state=None,\
            #                    solver='newton-cg',\
            #                    tol=0.0001, warm_start=False, n_jobs=-1)",
            "LinearSVC()",
            "MLPClassifier()",
            # "SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,\
            #     decision_function_shape='ovr', degree=0, gamma='auto', kernel='linear',\
            #     max_iter=-1, probability=True, random_state=None, shrinking=True,\
            #     tol=0.001, verbose=False)"
        ]

        i = 0
        for base_estimator in base_estimators:
            logging.warning(str(datetime.now()) + 'Start ' + model_params[i])
            try:
                model = OneVsRestClassifier(base_estimator, n_jobs=-1)
                # cross_val_f1 = Evaluator.evaluate_only_cross_val(model, x_all, y_all)
                # self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, model_params[i],
                #                                        "Word2Vec_CBOW", cross_val_f1)
                report, micro, macro, weighted = Evaluator.multi_label_report(model, x_all, y_all)
                self.__visualizer.save_metrics(self.__CLASSIFIER_NAME, model_params[i], "tfidf",
                                               report, micro, macro, weighted)
            except:
                logging.warning('Error on ' + model_params[i])
            logging.warning(str(datetime.now()) + 'End ' + model_params[i])
            i += 1

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors_cbow()
        y_all = self.__data_source.get_y_multi_label()

        # TODO here grid search

        base_estimators = [
            LogisticRegression(C=1.0, solver='sag', n_jobs=-1),
            LogisticRegression(n_jobs=-1),
            # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
            #                    intercept_scaling=True, max_iter=100, multi_class='ovr', penalty='l2', random_state=None,
            #                    solver='newton-cg',
            #                    tol=0.0001, warm_start=False, n_jobs=-1),
            LinearSVC(),
            MLPClassifier()
            # SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
            #     decision_function_shape='ovr', degree=0, gamma='auto', kernel='linear',
            #     max_iter=-1, probability=True, random_state=None, shrinking=True,
            #     tol=0.001, verbose=False)
        ]

        model_params = [
            "LogisticRegression(C=1.0, solver='sag')",
            'LogisticRegression()',
            # "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\
            #                    intercept_scaling=True, max_iter=100, multi_class='ovr', penalty='l2', random_state=None,\
            #                    solver='newton-cg',\
            #                    tol=0.0001, warm_start=False, n_jobs=-1)",
            "LinearSVC()",
            "MLPClassifier()",
            # "SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,\
            #     decision_function_shape='ovr', degree=0, gamma='auto', kernel='linear',\
            #     max_iter=-1, probability=True, random_state=None, shrinking=True,\
            #     tol=0.001, verbose=False)"
        ]

        i = 0
        for base_estimator in base_estimators:
            logging.warning(str(datetime.now()) + 'Start ' + model_params[i])
            try:
                model = OneVsRestClassifier(base_estimator, n_jobs=-1)
                # cross_val_f1 = Evaluator.evaluate_only_cross_val(model, x_all, y_all)
                # self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, model_params[i],
                #                                        "Word2Vec_CBOW", cross_val_f1)
                report, micro, macro, weighted = Evaluator.multi_label_report(model, x_all, y_all)
                self.__visualizer.save_metrics(self.__CLASSIFIER_NAME, model_params[i], "Word2Vec_CBOW",
                                               report, micro, macro, weighted)
            except:
                logging.warning('Error on ' + model_params[i])
            logging.warning(str(datetime.now()) + 'End ' + model_params[i])
            i += 1

    def make_use_d2v(self):
        x_all = self.__vectors_provider.get_d2v_vectors_dbow()
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
            'LogisticRegression()'
            "LinearSVC()",
            "MLPClassifier()"
        ]

        i = 0
        for base_estimator in base_estimators:
            logging.warning(str(datetime.now()) + 'Start ' + model_params[i])
            try:
                model = OneVsRestClassifier(base_estimator, n_jobs=-1)
                # cross_val_f1 = Evaluator.evaluate_only_cross_val(model, x_all, y_all)
                # self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, model_params[i],
                #                                        "Word2Vec_CBOW", cross_val_f1)
                report, micro, macro, weighted = Evaluator.multi_label_report(model, x_all, y_all)
                self.__visualizer.save_metrics(self.__CLASSIFIER_NAME, model_params[i], "Doc2Vec_DBOW",
                                               report, micro, macro, weighted)
            except:
                logging.warning('Error on ' + model_params[i])
            logging.warning(str(datetime.now()) + 'End ' + model_params[i])
            i += 1