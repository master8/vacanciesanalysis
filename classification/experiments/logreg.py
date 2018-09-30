import pandas as pd
from sklearn.linear_model import LogisticRegression

from classification.evaluation import Evaluator
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer


class LogisticRegressionExperiments:
    __ORIGINAL_DATASET_PATH = "../data/old/old_marked_vacancies_from_hh.csv"

    def __init__(self):
        self.__vectors_provider = VectorsProvider()

    def __read_original_dataset(self):
        return pd.read_csv(self.__ORIGINAL_DATASET_PATH, header=0)

    def make_use_tfidf(self):
        x_all = self.__vectors_provider.get_tfidf_vectors()
        y_all = self.__read_original_dataset().profession

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results("LogisticRegression", "model1", "TF-IDF",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__read_original_dataset().profession

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results("LogisticRegression", "model1", "Word2Vec",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)

    def make_use_w2v_with_tfidf(self):
        x_all = self.__vectors_provider.get_w2v_tfidf_vectors()
        y_all = self.__read_original_dataset().profession

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1 \
            = Evaluator.evaluate(model1, x_all, y_all)

        Visualizer.show_results("LogisticRegression", "model1", "Word2Vec&TF-IDF",
                                cross_val_accuracy, cross_val_f1,
                                train_accuracy, train_f1,
                                test_accuracy, test_f1)
