import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from classification.evaluation import Evaluator
from classification.source import DataSource
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer


class VotingExperiments:

    __CLASSIFIER_NAME = "VotingClassifier"

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

        estimators = []
        part1 = LogisticRegression(C=0.5, solver='liblinear')
        estimators.append(('logistic', part1))
        part2 = SVC(C=10, kernel='rbf')
        estimators.append(('svc', part2))
        part3 = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')
        estimators.append(('knn', part3))

        model1 = VotingClassifier(estimators)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "TF-IDF",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        estimators = []
        part1 = LogisticRegression(C=0.5, solver='liblinear')
        estimators.append(('logistic', part1))
        part2 = SVC(C=10, kernel='rbf')
        estimators.append(('svc', part2))
        part3 = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')
        estimators.append(('knn', part3))

        model1 = VotingClassifier(estimators)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "Word2Vec",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v_big(self):
        x_all = self.__vectors_provider.get_w2v_big_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        estimators = []
        part1 = LogisticRegression(C=0.5, solver='liblinear')
        estimators.append(('logistic', part1))
        part2 = SVC(C=10, kernel='rbf')
        estimators.append(('svc', part2))
        part3 = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')
        estimators.append(('knn', part3))

        model1 = VotingClassifier(estimators)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "Word2VecBig",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_w2v_with_tfidf(self):
        x_all = self.__vectors_provider.get_w2v_tfidf_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        estimators = []
        part1 = LogisticRegression(C=0.5, solver='liblinear')
        estimators.append(('logistic', part1))
        part2 = SVC(C=10, kernel='rbf')
        estimators.append(('svc', part2))
        part3 = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')
        estimators.append(('knn', part3))

        model1 = VotingClassifier(estimators)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "Word2Vec&TF-IDF",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_tfidf_wshingles(self):
        x_all = self.__vectors_provider.get_tfidf_wshingles_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        estimators = []
        part1 = LogisticRegression(C=0.5, solver='liblinear')
        estimators.append(('logistic', part1))
        part2 = SVC(C=10, kernel='rbf')
        estimators.append(('svc', part2))
        part3 = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')
        estimators.append(('knn', part3))

        model1 = VotingClassifier(estimators)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "TF-IDF&w-shingles",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)

    def make_use_tfidf_ngrams(self):
        x_all = self.__vectors_provider.get_tfidf_ngrams_vectors()
        y_all = self.__data_source.get_y()

        # TODO here grid search

        estimators = []
        part1 = LogisticRegression(C=0.5, solver='liblinear')
        estimators.append(('logistic', part1))
        part2 = SVC(C=10, kernel='rbf')
        estimators.append(('svc', part2))
        part3 = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')
        estimators.append(('knn', part3))

        model1 = VotingClassifier(estimators)

        cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_true, y_pred \
            = Evaluator.evaluate(model1, x_all, y_all)

        self.__visualizer.show_results(self.__CLASSIFIER_NAME, "model1", "TF-IDF&n-grams",
                                       cross_val_accuracy, cross_val_f1,
                                       train_accuracy, train_f1,
                                       test_accuracy, test_f1, y_true, y_pred)
