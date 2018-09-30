from classification.old.reading_data import read_tokenized_all_data
from classification.old.reading_data import read_all_data
from classification.stuff.classification_results import show_classification_results

import pandas as pd

from keras.preprocessing.text import Tokenizer

from sklearn.ensemble import GradientBoostingClassifier

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


def DatasetToNgrams(dataset,n):
    return [[' '.join(line[i:i+n]) for i in range(0,line.__len__()-(n-1))] for line in dataset]

size_ngrams = 3
ngramsDataset = DatasetToNgrams(hh_tokenized+sj_tokenized,size_ngrams)
print(ngramsDataset[:10])

tfidf, dictionary = get_texts_to_matrix(ngramsDataset)
tfidf_merged_ngrams = {'tfidf': tfidf, 'dictionary':dictionary}

# classification

x_all = tfidf_merged_ngrams['tfidf']
y_all = pd.concat([dataset_hh.profession, dataset_sj.profession])


model1 = GradientBoostingClassifier(loss='deviance', max_depth=3)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

model1.fit(x_train, y_train)
y_predict = model1.predict(x_test)

show_classification_results(y_test, y_predict, "tf-idf_n-grams_gr-boost_model1")
