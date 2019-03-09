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

import logging
from datetime import datetime


class Matcher:

    def __init__(self, n_similar_parts: int = 5, start_n: int = 0) -> None:
        super().__init__()

        self.__morph = pymorphy2.MorphAnalyzer()
        self.__cyrillic = u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.__allowed_characters = ascii_lowercase + digits + self.__cyrillic + whitespace
        self.__word2vec = Word2Vec.load('/home/mluser/prof/EduVacanciesStructuring/data/big_word2vec/big_word2vec_model_CBOW')
        self.__word2vec.wv.init_sims()
        self.__n_similar_parts = n_similar_parts
        self.__start_n = start_n

    def match_parts(self, vacancies_parts: pd.DataFrame, profstandards_parts: pd.DataFrame) -> pd.DataFrame:

        logging.warning('start match_parts')

        profstandards_parts['full_text'] = profstandards_parts['part_text'] + ' ' + \
                                           profstandards_parts['function_name'] + ' ' + \
                                           profstandards_parts['general_function_name']

        logging.warning('full_text done')

        vectorized_profstandards_parts = self.__preprocess(profstandards_parts, 'full_text')

        logging.warning('lemm standard done')

        vectorized_vacancies_parts = self.__preprocess(vacancies_parts, 'vacancy_part_text')

        logging.warning('lemm vacancies done')

        return self.__similarity(vectorized_vacancies_parts, vectorized_profstandards_parts)

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

    def __similarity(self, vacancies, standards, own=True):
        df_result = pd.DataFrame(columns=['enriched_text',
                                          'similarity', 'profstandard_part_id', 'vacancy_part_id'],
                                 index=None)
        own_code = [0]
        i = 0
        for index, sample in vacancies.iterrows():
            if own is True:
                labels = sample['profstandard_id']
                own_code = labels.split(',')
            similar_docs = self.__most_similar(sample['vectors'], standards, own_code)[['full_text', 'similarity',
                                                                                         'profstandard_part_id']]  # sc (близость нужна)
            similar_docs['vacancy_part_id'] = sample['vacancy_part_id']  # нужно
            similar_docs = similar_docs.rename(columns={
                'full_text': 'enriched_text',  # нужно
            })
            df_result = pd.concat([df_result, similar_docs], ignore_index=True)

            i = i + 1
            if i % 1000 == 0:
                logging.warning(str(datetime.now()) + ' done ' + str(i) + 'count ' + str(df_result.size))
                df_result.to_csv('../data/new/sim_result_mid' + str(self.__start_n) + '.csv', index=False)

        return df_result

    def __most_similar(self, infer_vector, vectorized_corpus, own_code=[0]):
        if own_code[0] != 0:
            df_sim = pd.DataFrame()
            for label in own_code:
                df_sim_label = vectorized_corpus[vectorized_corpus['profstandard_id'] == int(label)]
                df_sim = pd.concat([df_sim, df_sim_label], ignore_index=False)
        else:
            df_sim = vectorized_corpus

        df_sim['similarity'] = df_sim['vectors'].progress_apply(
            lambda v: cosine_similarity([infer_vector], [v.tolist()])[0, 0])
        # df_sim = df_sim.sort_values(by='similarity', ascending=False).head(n=self.__n_similar_parts)
        return df_sim

