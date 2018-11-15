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
            # LogisticRegression(n_jobs=-1),
            # SVC(),
            # KNeighborsClassifier(),
            # GaussianProcessClassifier(),
            # DecisionTreeClassifier(),
            # RandomForestClassifier(),
            # MLPClassifier(),
            # AdaBoostClassifier(),
            # GaussianNB(),
            # QuadraticDiscriminantAnalysis()
            # RidgeClassifier(),
            # SGDClassifier(n_jobs=-1),
            # Perceptron(n_jobs=-1),
            # PassiveAggressiveClassifier(n_jobs=-1),
            # BernoulliNB(),
            # LinearSVC(),
            BRkNNaClassifier(),
            BRkNNbClassifier(),
            MLkNN(),
            MLARAM(),
            BinaryRelevance(LogisticRegression(n_jobs=-1)),
            BinaryRelevance(LinearSVC()),
            BinaryRelevance(MLPClassifier()),
            ClassifierChain(LogisticRegression(n_jobs=-1)),
            ClassifierChain(LinearSVC()),
            ClassifierChain(MLPClassifier()),
            LabelPowerset(LogisticRegression(n_jobs=-1)),
            LabelPowerset(LinearSVC()),
            LabelPowerset(MLPClassifier())
        ]

        model_params = [
            # 'LogisticRegression()',
            # 'SVC()',
            # 'KNeighborsClassifier()',
            # 'GaussianProcessClassifier()',
            # 'DecisionTreeClassifier()',
            # 'RandomForestClassifier()',
            # 'MLPClassifier()',
            # 'AdaBoostClassifier()',
            # 'GaussianNB()',
            # 'QuadraticDiscriminantAnalysis()'
            # 'RidgeClassifier()',
            # 'SGDClassifier()',
            # 'Perceptron()',
            # 'PassiveAggressiveClassifier()',
            # 'BernoulliNB()',
            # 'LinearSVC()',
            'BRkNNaClassifier()',
            'BRkNNbClassifier()',
            'MLkNN()',
            'MLARAM()',
            'BinaryRelevance(LogisticRegression())',
            'BinaryRelevance(LinearSVC())',
            'BinaryRelevance(MLPClassifier())',
            'ClassifierChain(LogisticRegression())',
            'ClassifierChain(LinearSVC())',
            'ClassifierChain(MLPClassifier())',
            'LabelPowerset(LogisticRegression())',
            'LabelPowerset(LinearSVC())',
            'LabelPowerset(MLPClassifier())'
        ]

        i = 0
        for base_estimator in base_estimators:
            logging.warning('Start ' + model_params[i])
            try:
                # model = OneVsRestClassifier(base_estimator, n_jobs=-1)
                model = base_estimator
                cross_val_f1 = Evaluator.evaluate_only_cross_val(model, x_all, y_all)
                self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, model_params[i],
                                                       "tfidf", cross_val_f1)
            except:
                logging.warning('Error on ' + model_params[i])
            i += 1
            logging.warning('End ' + model_params[i])

        # model1 = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', n_jobs=-1), n_jobs=-1)
        # Evaluator.multi_label_predict_proba_tfidf(model1, x_all, y_all, data_source=self.__data_source)

        # cross_val_f1 = Evaluator.evaluate_only_cross_val(model1, x_all, y_all)
        #
        # self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, "LogisticRegression(C=1.0, solver='sag')",
        #                                "tfidf", cross_val_f1)

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__data_source.get_y_multi_label()

        # TODO here grid search

        base_estimators = [
            # LogisticRegression(n_jobs=-1),
            # SVC(),
            # KNeighborsClassifier(),
            # GaussianProcessClassifier(),
            # DecisionTreeClassifier(),
            # RandomForestClassifier(),
            # MLPClassifier(),
            # AdaBoostClassifier(),
            # GaussianNB(),
            # QuadraticDiscriminantAnalysis()
            # RidgeClassifier(),
            # SGDClassifier(n_jobs=-1),
            # Perceptron(n_jobs=-1),
            # PassiveAggressiveClassifier(n_jobs=-1),
            # BernoulliNB(),
            # LinearSVC(),
            # LinearDiscriminantAnalysis(),
            # BaggingClassifier(LogisticRegression(n_jobs=-1), n_jobs=-1),
            # BaggingClassifier(LinearSVC(), n_jobs=-1),
            # BaggingClassifier(MLPClassifier(), n_jobs=-1),
            # ExtraTreesClassifier(n_jobs=-1),
            # GradientBoostingClassifier(),
            #? VotingClassifier([LogisticRegression(n_jobs=-1), LinearSVC(), MLPClassifier()], n_jobs=-1),
            LogisticRegressionCV(n_jobs=-1),
            RidgeClassifierCV(),
            MultinomialNB(),
            # BRkNNaClassifier(),
            # BRkNNbClassifier(),
            # MLkNN(),
            # MLARAM(),
            # BinaryRelevance(LogisticRegression(n_jobs=-1)),
            # BinaryRelevance(LinearSVC()),
            # BinaryRelevance(MLPClassifier()),
            # ClassifierChain(LogisticRegression(n_jobs=-1)),
            # ClassifierChain(LinearSVC()),
            # ClassifierChain(MLPClassifier()),
            # LabelPowerset(LogisticRegression(n_jobs=-1)),
            # LabelPowerset(LinearSVC()),
            # LabelPowerset(MLPClassifier())
        ]

        model_params = [
            # 'LogisticRegression()',
            # 'SVC()',
            # 'KNeighborsClassifier()',
            # 'GaussianProcessClassifier()',
            # 'DecisionTreeClassifier()',
            # 'RandomForestClassifier()',
            # 'MLPClassifier()',
            # 'AdaBoostClassifier()',
            # 'GaussianNB()',
            # 'QuadraticDiscriminantAnalysis()'
            # 'RidgeClassifier()',
            # 'SGDClassifier()',
            # 'Perceptron()',
            # 'PassiveAggressiveClassifier()',
            # 'BernoulliNB()',
            # 'LinearSVC()',
            # 'LinearDiscriminantAnalysis()',
            # 'BaggingClassifier(LogisticRegression())',
            # 'BaggingClassifier(LinearSVC())',
            # 'BaggingClassifier(MLPClassifier())',
            # 'ExtraTreesClassifier()',
            # 'GradientBoostingClassifier()',
            # 'VotingClassifier([LogisticRegression(), LinearSVC(), MLPClassifier()])',
            'LogisticRegressionCV()',
            'RidgeClassifierCV()',
            'MultinomialNB()',
            # 'BRkNNaClassifier()',
            # 'BRkNNbClassifier()',
            # 'MLkNN()',
            # 'MLARAM()',
            # 'BinaryRelevance(LogisticRegression())',
            # 'BinaryRelevance(LinearSVC())',
            # 'BinaryRelevance(MLPClassifier())',
            # 'ClassifierChain(LogisticRegression())',
            # 'ClassifierChain(LinearSVC())',
            # 'ClassifierChain(MLPClassifier())',
            # 'LabelPowerset(LogisticRegression())',
            # 'LabelPowerset(LinearSVC())',
            # 'LabelPowerset(MLPClassifier())'
        ]

        i = 0
        for base_estimator in base_estimators:
            logging.warning('Start ' + model_params[i])
            try:
                # model = OneVsRestClassifier(base_estimator, n_jobs=-1)
                model = base_estimator
                cross_val_f1 = Evaluator.evaluate_only_cross_val(model, x_all, y_all)
                self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, model_params[i],
                                               "Word2Vec", cross_val_f1)
            except:
                logging.warning('Error on ' + model_params[i])
            i += 1
            logging.warning('End ' + model_params[i])



        # model1 = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', n_jobs=-1), n_jobs=-1)
        # Evaluator.multi_label_predict_proba_w2v(model1, x_all, y_all, data_source=self.__data_source)
        #
        # cross_val_f1 = Evaluator.evaluate_only_cross_val(model1, x_all, y_all)
        #
        # self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, "LogisticRegression(C=1.0, solver='sag')",
        #                                "Word2Vec", cross_val_f1)
