from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


class Visualizer:

    def __init__(self, corpus_name) -> None:
        super().__init__()
        self.__corpus_name = corpus_name

    def show_results(self, classifier_name, model_name, vec_method,
                     cross_val_accuracy, cross_val_f1,
                     train_accuracy, train_f1,
                     test_accuracy, test_f1, y_true, y_pred):
        print()
        print(classifier_name + " " + model_name + " " + vec_method)
        print()
        print("cross validation")
        print("accuracy:" + str(cross_val_accuracy))
        print("accuracy mean:" + str(cross_val_accuracy.mean()))
        print("f-score:" + str(cross_val_f1))
        print("f-score mean:" + str(cross_val_f1.mean()))
        print()
        print("Prediction train: ")
        print("accuracy: " + str(train_accuracy))
        print("f-score: " + str(train_f1))
        print()
        print("Prediction test: ")
        print("accuracy: " + str(test_accuracy))
        print("f-score: " + str(test_f1))
        print()

        file_path = "results/classification_results.csv"

        pd.read_csv(file_path, header=0).append({
            'date_time': datetime.now(),
            'classifier_name': classifier_name,
            'model_name': model_name,
            'vec_method': vec_method,
            'cross_val_accuracy': cross_val_accuracy.mean(),
            'cross_val_f1': cross_val_f1.mean(),
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'dataset': self.__corpus_name
        }, ignore_index=True).to_csv(file_path, index=False)

        self.__show_confusion_matrix(y_true, y_pred, classifier_name + '_' + vec_method)

    def __show_confusion_matrix(self, y_true, y_pred, name):

        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]

        matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(
            matrix, index=labels, columns=labels,
        )

        plt.figure(figsize=(20, 15))

        try:
            sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="BuPu")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('results/plots/' + self.__corpus_name + '/' + name + '.svg', format='svg')
