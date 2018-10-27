import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from classification.evaluation import Evaluator
from classification.source import DataSource
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer


class GradientBoostingExperiments:

    __CLASSIFIER_NAME = "GradientBoostingClassifier"

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

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(loss='deviance', max_depth=3)", "TF-IDF",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(loss='deviance', max_depth=3)", "Word2Vec",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v_big(self):
        x_all = self.__vectors_provider.get_w2v_big_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(loss='deviance', max_depth=3)", "Word2VecBig",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v_with_tfidf(self):
        x_all = self.__vectors_provider.get_w2v_tfidf_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(loss='deviance', max_depth=3)", "Word2Vec&TF-IDF",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_tfidf_wshingles(self):
        x_all = self.__vectors_provider.get_tfidf_wshingles_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(loss='deviance', max_depth=3)", "TF-IDF&w-shingles",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_tfidf_ngrams(self):
        x_all = self.__vectors_provider.get_tfidf_ngrams_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "(loss='deviance', max_depth=3)", "TF-IDF&n-grams",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)
