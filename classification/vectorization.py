import gensim
import pickle

import pandas
from keras_preprocessing.text import Tokenizer

from classification.tokenization import TokensProvider


_VECTORS_TFIDF_FILE_PATH = "prepared_data/vectors_tfidf_all_hh.pkl"
_VECTORS_TFIDF_WSHINGLES_FILE_PATH = "prepared_data/vectors_tfidf_wshingles_all_hh.pkl"
_VECTORS_TFIDF_NGRAMS_FILE_PATH = "prepared_data/vectors_tfidf_ngrams_all_hh.pkl"

_VECTORS_W2V_FILE_PATH = "prepared_data/vectors_w2v_all_hh.pkl"
_VECTORS_W2V_BIG_FILE_PATH = "prepared_data/vectors_w2v_big_all_hh.pkl"
_VECTORS__W2V_TFIDF_FILE_PATH = "prepared_data/vectors_w2v_tfidf_all_hh.pkl"


class Vectorizer:

    def __init__(self):
        self.__tokens_provider = TokensProvider()

    def vectorize_with_tfidf_ngrams(self):
        print("start tfidf ngrams vectorizing...")

        tokens = self.__tokens_provider.get_tokens()
        size_ngrams = 3
        ngrams = self.__DatasetToNgrams(tokens, size_ngrams)
        vectorized_tokens = self.__get_texts_to_matrix(ngrams)[0]

        outfile = open(_VECTORS_TFIDF_NGRAMS_FILE_PATH, 'wb')
        pickle.dump(vectorized_tokens, outfile)
        outfile.close()

        print("end tfidf ngrams vectorizing, vectors saved")

    # TODO порефакторить
    def __DatasetToNgrams(self, dataset, n):
        return [[' '.join(line[i:i + n]) for i in range(0, line.__len__() - (n - 1))] for line in dataset]

    def vectorize_with_tfidf_wshingles(self):
        print("start tfidf wshingles vectorizing...")

        tokens = self.__tokens_provider.get_tokens()
        size_shingles = 2
        wshingles = self.__DatasetToShingles(tokens, size_shingles)
        vectorized_tokens = self.__get_texts_to_matrix(wshingles)[0]

        outfile = open(_VECTORS_TFIDF_WSHINGLES_FILE_PATH, 'wb')
        pickle.dump(vectorized_tokens, outfile)
        outfile.close()

        print("end tfidf wshingles vectorizing, vectors saved")

    # TODO порефакторить
    def __DatasetToShingles(self, dataset, n):
        merged = [' '.join(d) for d in dataset]
        return [[''.join(line[i:i + n]) for i in range(0, line.__len__() - (n - 1))] for line in merged]

    def vectorize_with_w2v_tfidf(self):
        print("start w2v tfidf vectorizing...")

        tokens = self.__tokens_provider.get_tokens()
        w2v_model = gensim.models.Word2Vec(tokens, min_count=2, workers=2, iter=100, size=300, sg=0)

        results = self.__get_texts_to_matrix(tokens)
        tfidf_model = results[0]
        word_index = results[1]

        vectorized_tokens = self.__GetW2VTFIDFVectors(w2v_model, tfidf_model, word_index, tokens)

        outfile = open(_VECTORS__W2V_TFIDF_FILE_PATH, 'wb')
        pickle.dump(vectorized_tokens, outfile)
        outfile.close()

        print("end w2v tfidf vectorizing, vectors saved")

    # TODO порефакторить
    def __GetW2VTFIDFVectors(self, w2v_model, tfidf_model, tfidf_dict, vacancies):
        index = 0
        vectors = []
        for vacancy in vacancies:
            vector = self.__SentenceToAverageTfIdfWeightedVector(w2v_model.wv, vacancy, tfidf_model[index], tfidf_dict)
            vectors.append(vector)
            index += 1
        return vectors

    # TODO порефакторить
    def __SentenceToAverageTfIdfWeightedVector(self, wv, sentence, tfidf, dictionary):
        vectors = pandas.DataFrame()
        index = 0
        try:
            for word in sentence:
                if word not in dictionary.keys():
                    tf_idf = 0
                else:
                    word_index = dictionary[word]
                    tf_idf = tfidf[word_index]
                if word in wv.vocab:
                    vectors[index] = wv[word] * tf_idf
                index += 1
            vectors = vectors.transpose()
            vector = vectors.mean().values.tolist()
        except Exception:
            return []
        return vector

    def vectorize_with_w2v_big(self):
        print("start w2v big vectorizing...")

        tokens = self.__tokens_provider.get_tokens()
        # w2v_model = gensim.models.Word2Vec(tokens, min_count=2, workers=2, iter=100, size=300, sg=0)
        w2v_path = "prepared_data/all.norm-sz500-w10-cb0-it3-min5.w2v"
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True, unicode_errors='ignore')
        vectorized_tokens = [self.__SentenceToAverageWeightedVector(w2v_model.wv, vacancy) for vacancy in tokens]

        outfile = open(_VECTORS_W2V_BIG_FILE_PATH, 'wb')
        pickle.dump(vectorized_tokens, outfile)
        outfile.close()

        print("end w2v big vectorizing, vectors saved")

    def vectorize_with_w2v(self):
        print("start w2v vectorizing...")

        tokens = self.__tokens_provider.get_tokens()
        w2v_model = gensim.models.Word2Vec(tokens, min_count=2, workers=2, iter=100, size=300, sg=0)
        vectorized_tokens = [self.__SentenceToAverageWeightedVector(w2v_model.wv, vacancy) for vacancy in tokens]

        outfile = open(_VECTORS_W2V_FILE_PATH, 'wb')
        pickle.dump(vectorized_tokens, outfile)
        outfile.close()

        print("end w2v vectorizing, vectors saved")

    # TODO порефакторить
    def __SentenceToAverageWeightedVector(self, wv, sentence):
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

    def vectorize_with_tfidf(self):
        print("start tfidf vectorizing...")

        tokens = self.__tokens_provider.get_tokens()
        vectorized_tokens = self.__get_texts_to_matrix(tokens)[0]

        outfile = open(_VECTORS_TFIDF_FILE_PATH, 'wb')
        pickle.dump(vectorized_tokens, outfile)
        outfile.close()

        print("end tfidf vectorizing, vectors saved")

    # TODO порефакторить
    def __get_texts_to_matrix(self, texts, max_features=0):
        tokenizer = Tokenizer(split=" ", lower=True)
        if max_features != 0:
            tokenizer = Tokenizer(split=" ", lower=True, num_words=max_features, char_level='True')

        tokenizer.fit_on_texts(texts)
        matrix_tfidf = tokenizer.texts_to_matrix(texts=texts, mode='tfidf')
        return matrix_tfidf, tokenizer.word_index


class VectorsProvider:

    def get_tfidf_vectors(self):
        file = open(_VECTORS_TFIDF_FILE_PATH, 'rb')
        vectors = pickle.load(file)
        file.close()
        return vectors

    def get_w2v_vectors(self):
        file = open(_VECTORS_W2V_FILE_PATH, 'rb')
        vectors = pickle.load(file)
        file.close()
        return vectors

    def get_w2v_big_vectors(self):
        file = open(_VECTORS_W2V_BIG_FILE_PATH, 'rb')
        vectors = pickle.load(file)
        file.close()
        return vectors

    def get_w2v_tfidf_vectors(self):
        file = open(_VECTORS__W2V_TFIDF_FILE_PATH, 'rb')
        vectors = pickle.load(file)
        file.close()
        return vectors

    def get_tfidf_wshingles_vectors(self):
        file = open(_VECTORS_TFIDF_WSHINGLES_FILE_PATH, 'rb')
        vectors = pickle.load(file)
        file.close()
        return vectors

    def get_tfidf_ngrams_vectors(self):
        file = open(_VECTORS_TFIDF_NGRAMS_FILE_PATH, 'rb')
        vectors = pickle.load(file)
        file.close()
        return vectors
