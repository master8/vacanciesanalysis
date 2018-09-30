from classification.old.reading_data import read_tokenized_all_data
from classification.old.reading_data import read_all_data
from classification.stuff.classification_results import show_classification_results

import pandas
import pandas as pd
import gensim

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


dataset_hh, dataset_sj = read_all_data()
hh_tokenized, sj_tokenized = read_tokenized_all_data()


# into vectors


def W2VStats(name, sentences):
    counter = [s.__len__() for s in sentences]
    print('Total n/o words in ' + name + ' : ' + str(sum(counter)))

# W2VStats('merged', hh_tokenized + sj_tokenized)
w2v_hh_sj = gensim.models.Word2Vec(hh_tokenized + sj_tokenized, min_count=2, workers=2, iter=100, size=300, sg=0)
# w2v_hh_sj.save("../../data/old/old_vectorized_vacancies_all.w2v")
# w2v_hh_sj = gensim.models.Word2Vec.load("../../data/old/old_vectorized_vacancies_all.w2v")


def SentenceToAverageWeightedVector(wv, sentence):
    vectors = pandas.DataFrame()
    index = 0
    try:
        for word in sentence:
            if word in wv.vocab:
                vectors[index] = wv[word]
            index += 1
        vectors = vectors.transpose()
        vector = vectors.mean().values.tolist()
    except Exception:
        return []
    return vector


vectors_w2v_hh_merged = [SentenceToAverageWeightedVector(w2v_hh_sj.wv, vacancy) for vacancy in hh_tokenized]
vectors_w2v_sj_merged = [SentenceToAverageWeightedVector(w2v_hh_sj.wv, vacancy) for vacancy in sj_tokenized]


# classification


x_all = vectors_w2v_sj_merged + vectors_w2v_hh_merged
y_all = pd.concat([dataset_sj.profession, dataset_hh.profession])


# w2c tf-idf 0.89
# model1 = LogisticRegression(C=0.5, solver='liblinear')

# w2v
# Testing accuracy: 0.9253731343283582
# Testing F1 score: 0.9246134547385705
model2 = LogisticRegression(C=1.0, solver='sag')

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3)

model2.fit(x_train, y_train)
y_predict = model2.predict(x_test)

show_classification_results(y_test, y_predict, "w2v_log_reg_model2")