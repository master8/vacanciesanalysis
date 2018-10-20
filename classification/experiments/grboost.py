import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from classification.evaluation import Evaluator
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer


class GradientBoostingExperiments:
    # __ORIGINAL_DATASET_PATH = "../data/old/old_marked_vacancies_from_hh.csv"
    __ORIGINAL_DATASET_PATH = "../data/old/marked_vacancies_hh_sz100_201018.csv"
    __CLASSIFIER_NAME = "GradientBoostingClassifier"

    def __init__(self):
        self.__vectors_provider = VectorsProvider()

    def __read_y_for_original_dataset(self):
        return pd.read_csv(self.__ORIGINAL_DATASET_PATH, header=0).standard_mark

    def make_use_tfidf(self):
        x_all = self.__vectors_provider.get_tfidf_vectors()
        y_all = self.__read_y_for_original_dataset()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "TF-IDF",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__read_y_for_original_dataset()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "Word2Vec",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)

    def make_use_w2v_big(self):
        x_all = self.__vectors_provider.get_w2v_big_vectors()
        y_all = self.__read_y_for_original_dataset()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "Word2VecBig",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)

    def make_use_w2v_with_tfidf(self):
        x_all = self.__vectors_provider.get_w2v_tfidf_vectors()
        y_all = self.__read_y_for_original_dataset()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "Word2Vec&TF-IDF",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)

    def make_use_tfidf_wshingles(self):
        x_all = self.__vectors_provider.get_tfidf_wshingles_vectors()
        y_all = self.__read_y_for_original_dataset()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "TF-IDF&w-shingles",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)

    def make_use_tfidf_ngrams(self):
        x_all = self.__vectors_provider.get_tfidf_ngrams_vectors()
        y_all = self.__read_y_for_original_dataset()

        # TODO here grid search

        model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "TF-IDF&n-grams",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)
