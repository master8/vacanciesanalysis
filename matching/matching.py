import pandas as pd

import codecs
import os
import pymorphy2
from string import ascii_lowercase, digits, whitespace
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import gensim
from gensim.models import Word2Vec

from tqdm import tqdm
tqdm.pandas()


class Matcher:

    def __init__(self) -> None:
        super().__init__()

        self.__morph = pymorphy2.MorphAnalyzer()
        self.__cyrillic = u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.__allowed_characters = ascii_lowercase + digits + self.__cyrillic + whitespace
        self.__word2vec = Word2Vec.load('/home/mluser/prof/EduVacanciesStructuring/data/big_word2vec/big_word2vec_model_CBOW')
        self.__word2vec.wv.init_sims()

    def match_parts(self, vacancies_parts: pd.DataFrame, profstandards_parts: pd.DataFrame) -> pd.DataFrame:

        profstandards_parts['full_text'] = profstandards_parts['part_text'] + ' ' + \
                                           profstandards_parts['function_name'] + ' ' + \
                                           profstandards_parts['general_function_name']

        vectorized_profstandards_parts = self.__preprocess(profstandards_parts, 'full_text')
        vectorized_vacancies_parts = self.__preprocess(vacancies_parts, 'vacancy_part_text')

        return vectorized_profstandards_parts, vectorized_vacancies_parts

    def __preprocess(self, copus: pd.DataFrame, columns_name: str) -> pd.DataFrame:

        copus['processed_text'] = copus[columns_name].progress_apply(
            lambda text: self.__process_text(str(text))['lemmatized_text_pos_tags'])
        return self.__get_vectorized_avg_w2v_corpus(copus, self.__word2vec.wv)

    def __process_text(self, full_text, filter_pos=("PREP", "NPRO", "CONJ")):
        '''Process a single text and return a processed version
        '''
        single_line_text = full_text.replace('\n', ' ')
        preprocessed_text = self.__complex_preprocess(single_line_text)
        lemmatized_text, lemmatized_text_pos_tags = self.__lemmatize(preprocessed_text, filter_pos=filter_pos)

        return {"full_text": full_text,
                "single_line_text": single_line_text,
                "preprocessed_text": preprocessed_text,
                "lemmatized_text": lemmatized_text,
                "lemmatized_text_pos_tags": lemmatized_text_pos_tags}

    def __complex_preprocess(self, text, additional_allowed_characters="+#"):
        return ''.join(
            [character if character in set(self.__allowed_characters + additional_allowed_characters) else ' ' for character in
             text.lower()]).split()

    def __lemmatize(self, tokens, filter_pos):
        '''Produce normal forms for russion words using pymorphy2
        '''
        lemmas = []
        tagged_lemmas = []
        for token in tokens:
            parsed_token = self.__morph.parse(token)[0]
            norm = parsed_token.normal_form
            pos = parsed_token.tag.POS
            if pos is not None:
                if pos not in filter_pos:
                    lemmas.append(norm)
                    tagged_lemmas.append(norm + "_" + pos)
            else:
                lemmas.append(token)
                tagged_lemmas.append(token + "_")

        return lemmas, tagged_lemmas

    def __get_vectorized_avg_w2v_corpus(self, corpus, model):
        documents = corpus['processed_text'].tolist()

        document_vectors = [self.__word_averaging(model, document) for document in documents]
        clean_corpus = corpus
        clean_corpus['vectors'] = pd.Series(document_vectors).values

        return clean_corpus

    def __word_averaging(self, wv, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(wv.vector_size, )

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

