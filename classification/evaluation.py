import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, KFold

from classification.source import DataSource


class Evaluator:

    @staticmethod
    def evaluate(model, x_all, y_all):
        cross_val_accuracy = cross_val_score(estimator=model, X=x_all, y=y_all, scoring='accuracy', cv=5, n_jobs=-1)
        cross_val_f1 = cross_val_score(estimator=model, X=x_all, y=y_all, scoring='f1_weighted', cv=5, n_jobs=-1)

        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

        model.fit(x_train, y_train)
        y_predict_train = model.predict(x_train)

        train_accuracy = accuracy_score(y_train, y_predict_train)
        train_f1 = f1_score(y_train, y_predict_train, average='weighted')

        y_predict_test = model.predict(x_test)

        test_accuracy = accuracy_score(y_test, y_predict_test)
        test_f1 = f1_score(y_test, y_predict_test, average='weighted')

        return cross_val_accuracy, cross_val_f1, train_accuracy, train_f1, test_accuracy, test_f1, y_test, y_predict_test

    @staticmethod
    def evaluate_only_cross_val(model, x_all, y_all):
        cross_val_f1 = cross_val_score(estimator=model, X=x_all, y=y_all, scoring='f1_weighted', cv=5, n_jobs=-1)
        return cross_val_f1

    @staticmethod
    def cross_probabilities(model, x_all, y_all, data_source: DataSource, method: str):
        proba = cross_val_predict(model, x_all, y_all, cv=KFold(n_splits=5, shuffle=True), method='predict_proba', n_jobs=-1)

        proba = pd.DataFrame(proba)
        temp = pd.DataFrame()

        temp['true_mark'] = y_all
        temp['true_mark_index'] = temp.true_mark - 1
        temp.loc[temp.true_mark > 13, 'true_mark_index'] = temp.true_mark_index - 1

        proba_true = []
        for (i, row) in temp.iterrows():
            proba_true.append(proba.iloc[i][int(row.true_mark_index)])

        proba_true_column = 'proba_true_' + method
        pred_mark_column = 'pred_mark_' + method
        proba_pred_column = 'proba_pred_' + method

        proba_true = pd.Series(proba_true)
        temp[proba_true_column] = proba_true
        temp = temp.drop(columns=['true_mark_index'])
        temp[pred_mark_column] = proba.idxmax(axis=1)
        temp[pred_mark_column] = temp[pred_mark_column] + 1
        temp.loc[temp[pred_mark_column] > 12, pred_mark_column] = temp[pred_mark_column] + 1
        temp[proba_pred_column] = proba.max(axis=1)

        co = data_source.get_corpus()
        # co['true_mark'] = temp['true_mark']
        co[proba_true_column] = temp[proba_true_column]
        co[pred_mark_column] = temp[pred_mark_column]
        co[proba_pred_column] = temp[proba_pred_column]
        data_source.save_corpus(co)
