import pickle

from keras_preprocessing.text import Tokenizer

from classification.tokenization import TokensProvider


_VECTORS_TFIDF_FILE_PATH = "prepared_data/vectors_tfidf_all_hh.pkl"
_VECTORS_W2V_FILE_PATH = "prepared_data/vectors_w2v_all_hh.pkl"
_VECTORS__W2V_TFIDF_FILE_PATH = "prepared_data/vectors_w2v_tfidf_all_hh.pkl"


class Vectorizer:

    def __init__(self):
        self.__tokens_provider = TokensProvider()

    def vectorize_with_tfidf(self):
        print("start tfidf vectorizing...")

        vectorized_tokens = self.__get_texts_to_matrix(self.__tokens_provider.get_tokens())

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
        return matrix_tfidf


class VectorsProvider:

    def get_tfidf_vectors(self):
        file = open(_VECTORS_TFIDF_FILE_PATH, 'rb')
        vectors = pickle.load(file)
        file.close()
        return vectors
