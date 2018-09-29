import pickle

from nltk.corpus import stopwords
from pymystem3 import Mystem

import pandas as pd

# pymystem3.mystem.MYSTEM_DIR = "/home/mluser/anaconda3/envs/master8_env/.local/bin"
# pymystem3.mystem.MYSTEM_BIN = "/home/mluser/anaconda3/envs/master8_env/.local/bin/mystem"


def tokenize_sentences_lemmatized(rawSentences):
    print('LEMMATIZED total = ' + str(rawSentences.__len__()))
    sentences = []
    m = Mystem()
    index = 0
    for c in rawSentences:
        tokenized_sents = m.lemmatize(c)
        cleaned_set = []
        for tokenized in tokenized_sents:
            if tokenized == "":
                break
            tokenized = tokenized.lower()
            if tokenized in stopwords.words('russian'):
                continue

            token = tokenized[0]
            if (token >= 'а' and token <= 'я'):
                cleaned_set.append(tokenized)
            elif ((token >= 'а' and token <= 'я') or (token >= 'a' and token <= 'z')):
                cleaned_set.append(tokenized)

        if cleaned_set.__len__() > 0:
            sentences.append(cleaned_set)
        if index % 1000 == 0:
            print(index)
        index += 1
    return sentences


dataset_hh = pd.read_csv("../../data/old/old_marked_vacancies_from_hh.csv", header=0)[:10]
tokenized_hh = tokenize_sentences_lemmatized(dataset_hh.requirements)

filename_hh = "../../data/old/old_tokenized_vacancies_from_hh.pkl"
outfile_hh = open(filename_hh, 'wb')
pickle.dump(tokenized_hh, outfile_hh)
outfile_hh.close()


dataset_sj = pd.read_csv("../../data/old/old_marked_vacancies_from_sj.csv", header=0)[:10]
tokenized_sj = tokenize_sentences_lemmatized(dataset_sj.requirements)

filename_sj = "../../data/old/old_tokenized_vacancies_from_sj.pkl"
outfile_sj = open(filename_sj, 'wb')
pickle.dump(tokenized_sj, outfile_sj)
outfile_sj.close()