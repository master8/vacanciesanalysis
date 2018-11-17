import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import MultiLabelBinarizer

from classification.marking import clean_label
from classification.source import DataSource
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


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
        binarizer = MultiLabelBinarizer()
        y_all = binarizer.fit_transform(y_all)
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

    @staticmethod
    def check_and_add_label(mark_proba, mark: str):
        results = ''

        if mark_proba > 0.5:
            results += mark + ','

        return results

    @staticmethod
    def multi_label_report(model, x_all, y_all):
        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19',
                   '20', '21']
        temp = pd.DataFrame(y_all, columns=['labels'])

        binarizer = MultiLabelBinarizer(classes=classes)
        y_all = binarizer.fit_transform(y_all)
        x_all = np.array(x_all)

        kf = KFold(n_splits=5, shuffle=True)

        predict = []
        for train, test in kf.split(x_all, y_all):
            x_train = x_all[train]
            y_train = y_all[train]

            x_test = x_all[test]
            y_test = y_all[test]

            m = clone(model)
            m.fit(x_train, y_train)
            predict.append(pd.DataFrame(m.predict(x_test), index=test, columns=classes))

        y_pred = pd.concat(predict)
        results = pd.merge(temp, y_pred, left_index=True, right_index=True, how='outer')
        y_pred = results.drop(columns='labels')
        print(classification_report(y_all, y_pred, target_names=classes))
        print('macro ' + str(precision_recall_fscore_support(y_all, y_pred, average='macro')))
        print('micro ' + str(precision_recall_fscore_support(y_all, y_pred, average='micro')))
        print('weighted ' + str(precision_recall_fscore_support(y_all, y_pred, average='weighted')))


    @staticmethod
    def multi_label_predict_proba_w2v(model, x_all, y_all, data_source: DataSource):

        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19','20', '21']
        temp = pd.DataFrame(y_all, columns=['labels'])

        binarizer = MultiLabelBinarizer(classes=classes)
        y_all = binarizer.fit_transform(y_all)
        x_all = np.array(x_all)

        kf = KFold(n_splits=5, shuffle=True)

        proba = []
        for train, test in kf.split(x_all, y_all):
            x_train = x_all[train]
            y_train = y_all[train]

            x_test = x_all[test]
            y_test = y_all[test]

            m = clone(model)
            m.fit(x_train, y_train)
            proba.append(pd.DataFrame(m.predict_proba(x_test), index=test, columns=classes))

        proba = pd.concat(proba)

        results = pd.merge(temp, proba, left_index=True, right_index=True, how='outer')

        y_true = pd.DataFrame(y_all, columns=classes)
        temp = pd.DataFrame(index=results.index)

        for mark in classes:
            temp['diff_' + mark] = y_true[mark] - results[mark]
            temp['diff_' + mark] = temp['diff_' + mark].abs()

        temp['max_wrong'] = temp.max(axis=1)

        results['pred_labels_w2v'] = ''

        for mark in classes:
            results.pred_labels_w2v = results.pred_labels_w2v + results[mark].apply(Evaluator.check_and_add_label, mark=mark)

        results.pred_labels_w2v = results.pred_labels_w2v.apply(clean_label)

        co = data_source.get_corpus()
        co['pred_labels_w2v'] = results.pred_labels_w2v
        co['max_wrong'] = temp['max_wrong']

        for mark in classes:
            co['m' + mark] = results[mark]

        data_source.save_corpus(co)

        temp = co.sort_values(by='max_wrong', ascending=False)
        temp = temp[temp.max_wrong > 0.5]
        temp = temp.drop(columns=['created_at', 'area_id', 'area_name', 'employer_id',
                                  'employer_name', 'employer_url', 'employment_id', 'experience_id',
                                  'schedule_id', 'description', 'specializations', 'key_skills',
                                  'salary_f_t_g_c', 'alternate_url', 'duties', 'requirements', 'standard_mark',
                                  'proba_true_w2v', 'pred_mark_w2v', 'proba_pred_w2v', 'proba_true_tfidf',
                                  'pred_mark_tfidf', 'proba_pred_tfidf'])

        data_source.save_confusion(temp)

    @staticmethod
    def multi_label_predict_proba_tfidf(model, x_all, y_all, data_source: DataSource):

        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19',
                   '20', '21']
        temp = pd.DataFrame(y_all, columns=['labels'])

        binarizer = MultiLabelBinarizer(classes=classes)
        y_all = binarizer.fit_transform(y_all)
        x_all = np.array(x_all)

        kf = KFold(n_splits=5, shuffle=True)

        proba = []
        for train, test in kf.split(x_all, y_all):
            x_train = x_all[train]
            y_train = y_all[train]

            x_test = x_all[test]
            y_test = y_all[test]

            m = clone(model)
            m.fit(x_train, y_train)
            proba.append(pd.DataFrame(m.predict_proba(x_test), index=test, columns=classes))

        proba = pd.concat(proba)

        results = pd.merge(temp, proba, left_index=True, right_index=True, how='outer')
        results['pred_labels_tfidf'] = ''

        for mark in classes:
            results.pred_labels_tfidf = results.pred_labels_tfidf + results[mark].apply(Evaluator.check_and_add_label,
                                                                                    mark=mark)

        results.pred_labels_tfidf = results.pred_labels_tfidf.apply(clean_label)

        co = data_source.get_corpus()
        co['pred_labels_tfidf'] = results.pred_labels_tfidf

        data_source.save_corpus(co)
