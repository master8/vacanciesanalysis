import pandas as pd
from sklearn.linear_model import LogisticRegression

from classification.evaluation import Evaluator
from classification.source import DataSource
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer


class LogisticRegressionExperiments:

    __CLASSIFIER_NAME = "LogisticRegression"

    def __init__(self,
                 data_source: DataSource,
                 vectors_provider: VectorsProvider,
                 visualizer: Visualizer):
        self.__data_source = data_source
        self.__vectors_provider = vectors_provider
        self.__visualizer = visualizer

    def make_use_tfidf(self):
        x_all = self.__vectors_provider.get_tfidf_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(C=1.0, solver='sag')", "TF-IDF",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(C=1.0, solver='sag')", "Word2Vec",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v_old(self):

        """ Обучает старый датасет с использованием w2v обученного на новом """

        x_all = self.__vectors_provider.get_w2v_old_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(C=1.0, solver='sag')", "Word2VecNewOld",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v_big(self):
        x_all = self.__vectors_provider.get_w2v_big_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(C=1.0, solver='sag')", "Word2VecBig",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v_with_tfidf(self):
        x_all = self.__vectors_provider.get_w2v_tfidf_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(C=1.0, solver='sag')", "Word2Vec&TF-IDF",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_tfidf_wshingles(self):
        x_all = self.__vectors_provider.get_tfidf_wshingles_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(C=1.0, solver='sag')", "TF-IDF&w-shingles",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_tfidf_ngrams(self):
        x_all = self.__vectors_provider.get_tfidf_ngrams_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(C=1.0, solver='sag')", "TF-IDF&n-grams",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)
