import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

from classification.vectorization import VectorsProvider
from classification.visualisation.classification_results import show_classification_results


class LogisticRegressionExperiments:

    __ORIGINAL_DATASET_PATH = "../data/old/old_marked_vacancies_from_hh.csv"

    def __init__(self):
        self.__vectors_provider = VectorsProvider()

    def __read_original_dataset(self):
        return pd.read_csv(self.__ORIGINAL_DATASET_PATH, header=0)

    def __cross_validation(self, model, x_all, y_all):
        print("cross validation")

        accuracy = cross_val_score(estimator=model, X=x_all, y=y_all, scoring='accuracy', cv=5)
        print("accuracy:" + str(accuracy))
        print("accuracy mean:" + str(accuracy.mean()))
        print()

        f1 = cross_val_score(estimator=model, X=x_all, y=y_all, scoring='f1_weighted', cv=5)
        print("f-score:" + str(f1))
        print("f-score mean:" + str(f1.mean()))
        print()

    def make_use_tfidf(self):
        x_all = self.__vectors_provider.get_tfidf_vectors()
        y_all = self.__read_original_dataset().profession

        # TODO here grid search

        model1 = LogisticRegression(C=1.0, solver='sag')

        self.__cross_validation(model1, x_all, y_all)

        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

        model1.fit(x_train, y_train)
        y_predict_train = model1.predict(x_train)

        print("Prediction train: ")
        print("accuracy: " + str(accuracy_score(y_train, y_predict_train)))
        print("f-score: " + str(f1_score(y_train, y_predict_train, average='weighted')))
        print()

        y_predict_test = model1.predict(x_test)

        print("Prediction train: ")
        print("accuracy: " + str(accuracy_score(y_test, y_predict_test)))
        print("f-score: " + str(f1_score(y_test, y_predict_test, average='weighted')))
        print()

        # show_classification_results(y_test, y_predict, "log_reg_tfidf_model1")
