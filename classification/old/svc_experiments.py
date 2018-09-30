import pickle

from classification.old.reading_data import read_tokenized_all_data
from classification.old.reading_data import read_all_data
from classification.stuff.classification_results import show_classification_results

import pandas as pd

from keras.preprocessing.text import Tokenizer

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split


dataset_hh, dataset_sj = read_all_data()
hh_tokenized, sj_tokenized = read_tokenized_all_data()

# into vectors

def get_texts_to_matrix(texts, max_features=0):
    tokenizer = Tokenizer(split=" ", lower=True)
    if max_features != 0:
        tokenizer = Tokenizer(split=" ", lower=True, num_words=max_features,char_level='True')

    tokenizer.fit_on_texts(texts)
    matrix_tfidf = tokenizer.texts_to_matrix(texts=texts, mode='tfidf')
    print('Количество текстов:', matrix_tfidf.shape[0])
    print('Количество токенов:', matrix_tfidf.shape[1])
    return matrix_tfidf, tokenizer.word_index


# tfidf, dictionary = get_texts_to_matrix(hh_tokenized+sj_tokenized)
# tfidf_hh_sj = {'tfidf': tfidf, 'dictionary':dictionary}
file = open("../../data/old/old_vec_tf-idf_vacancies_all.tfidf", 'rb')
x_all = pickle.load(file)
file.close()


# classification

# x_all = tfidf_hh_sj['tfidf']
y_all = pd.concat([dataset_hh.profession, dataset_sj.profession])

# w2c tf-idf 0.7
# model1 = SVC(C=10, kernel='rbf')


# tf-idf
# Testing accuracy: 0.908315565031983
# Testing F1 score: 0.9099553513175059
model2 = SVC(C=1, kernel='linear', probability=True)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

model2.fit(x_train, y_train)
y_predict = model2.predict(x_test)

show_classification_results(y_test, y_predict, "tf_idf_svc_model2")