from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split


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
