from classification.old.reading_data import read_tokenized_all_data
from classification.old.reading_data import read_all_data
from classification.stuff.classification_results import show_classification_results

import pandas as pd

from keras.preprocessing.text import Tokenizer

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

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

def DatasetToShingles(dataset, n):
    merged = [' '.join(d) for d in dataset]
    return [[''.join(line[i:i+n]) for i in range(0,line.__len__()-(n-1))] for line in merged]

size_shingles = 2
shinglesDataset = DatasetToShingles(hh_tokenized+sj_tokenized,size_shingles)

tfidf, dictionary = get_texts_to_matrix(shinglesDataset)
tfidf_merged_shingles = {'tfidf': tfidf, 'dictionary':dictionary}


# classification

x_all = tfidf_merged_shingles['tfidf']
y_all = pd.concat([dataset_hh.profession, dataset_sj.profession])


estimatorsTFIDFW2V = []
model1 = LogisticRegression(C =0.5,solver = 'liblinear')
estimatorsTFIDFW2V.append(('logistic', model1))
model2 = SVC(C = 10, kernel = 'rbf')
estimatorsTFIDFW2V.append(('svc', model2))
model3 = KNeighborsClassifier(algorithm='auto',metric= 'minkowski',weights= 'distance')
estimatorsTFIDFW2V.append(('knn', model3))


# Testing accuracy: 0.8699360341151386
# Testing F1 score: 0.870437507538225
model1 = VotingClassifier(estimatorsTFIDFW2V)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

model1.fit(x_train, y_train)
y_predict = model1.predict(x_test)

show_classification_results(y_test, y_predict, "tf-idf_w-shingles_ensemble_model1")
