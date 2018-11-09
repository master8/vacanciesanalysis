from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from classification.evaluation import Evaluator
from classification.source import DataSource
from classification.vectorization import VectorsProvider
from classification.visualisation import Visualizer


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

        binarizer = MultiLabelBinarizer()
        y_all = binarizer.fit_transform(y_all)

        # TODO here grid search

        model1 = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', n_jobs=-1), n_jobs=-1)

        cross_val_f1 = Evaluator.evaluate_only_cross_val(model1, x_all, y_all)

        self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, "LogisticRegression(C=1.0, solver='sag')",
                                       "tfidf", cross_val_f1)

    def make_use_w2v(self):
        x_all = self.__vectors_provider.get_w2v_vectors()
        y_all = self.__data_source.get_y_multi_label()

        binarizer = MultiLabelBinarizer()
        y_all = binarizer.fit_transform(y_all)

        # TODO here grid search

        model1 = OneVsRestClassifier(LogisticRegression(C=1.0, solver='sag', n_jobs=-1), n_jobs=-1)

        cross_val_f1 = Evaluator.evaluate_only_cross_val(model1, x_all, y_all)

        self.__visualizer.show_results_briefly(self.__CLASSIFIER_NAME, "LogisticRegression(C=1.0, solver='sag')",
                                       "Word2Vec", cross_val_f1)
