import pickle
import logging
import re
from datetime import datetime

from nltk.corpus import stopwords
from pymystem3 import Mystem

from classification.source import DataSource


_TOKENS_BASE_PATH = "prepared_data/"
_TOKENS_FILE_NAME = "/tokens.pkl"


class Tokenizer:

    def __init__(self, data_source: DataSource,
                 corpus_name) -> None:
        super().__init__()
        self.__data_source = data_source
        self.__corpus_name = corpus_name

    def tokenize(self):
        print("start tokenizing...")
        logging.warning(str(datetime.now()) + " start tokenizing...")

        tokenized_requirements = self.__tokenize_sentences_lemmatized(self.__data_source.get_x())

        outfile = open(_TOKENS_BASE_PATH + self.__corpus_name + _TOKENS_FILE_NAME, 'wb')
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
            logging.warning(str(datetime.now()) + " tokinizeing " + str(index))
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

    def __init__(self, corpus_name) -> None:
        super().__init__()
        self.__corpus_name = corpus_name

    def get_tokens(self):
        file = open(_TOKENS_BASE_PATH + self.__corpus_name + _TOKENS_FILE_NAME, 'rb')
        tokens = pickle.load(file)
        file.close()
        return tokens
