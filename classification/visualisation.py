from datetime import datetime

import pandas as pd


class Visualizer:

    @staticmethod
    def show_results(classifier_name, model_name, vec_method,
                     cross_val_accuracy, cross_val_f1,
                     train_accuracy, train_f1,
                     test_accuracy, test_f1):
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
            'dataset': 'hh_sz100'
        }, ignore_index=True).to_csv(file_path, index=False)
