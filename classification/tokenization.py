import pickle
import logging
import re
from datetime import datetime

from nltk.corpus import stopwords
from pymystem3 import Mystem

import pandas as pd


_TOKENS_FILE_PATH = "prepared_data/old_tokens_all_hh.pkl"
# _TOKENS_FILE_PATH = "prepared_data/tokens_all_hh.pkl"


class Tokenizer:

    # __ORIGINAL_DATASET_PATH = "../data/old/old_marked_vacancies_from_hh.csv"
    __ORIGINAL_DATASET_PATH = "../data/new/vacancies_hh_all_051018.csv"

    def __read_original_dataset(self):
        return pd.read_csv(self.__ORIGINAL_DATASET_PATH, header=0, sep='|')

    def clean(self, str):
        pattern = re.compile('<.*?>')
        return pattern.sub('', str)

    def tokenize(self):
        print("start tokenizing...")
        logging.warning(str(datetime.now()) + " start tokenizing...")

        original_dataset = self.__read_original_dataset()
        tokenized_requirements = self.__tokenize_sentences_lemmatized(original_dataset.description)

        outfile = open(_TOKENS_FILE_PATH, 'wb')
        pickle.dump(tokenized_requirements, outfile)
        outfile.close()

        logging.warning(str(datetime.now()) + " end tokenizing, tokens saved")
        print("end tokenizing, tokens saved")

    # TODO порефакторить
    def __tokenize_sentences_lemmatized(self, rawSentences):
        sentences = []
        m = Mystem()
        index = 0
        for c in rawSentences:
            c = self.clean(c)
            logging.warning(str(datetime.now()) + " tokinizeing " + str(index) + " " + c[:50])
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
            index += 1

        return sentences


class TokensProvider:

    def get_tokens(self):
        file = open(_TOKENS_FILE_PATH, 'rb')
        tokens = pickle.load(file)
        file.close()
        return tokens
