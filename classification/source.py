import pandas as pd


class DataSource:
    def __init__(self,
                 corpus_name,
                 x_column_name,
                 y_column_name) -> None:
        super().__init__()
        self.__corpus_name = corpus_name
        self.__x_column_name = x_column_name
        self.__y_column_name = y_column_name

        self.__corpus: pd.DataFrame = None

    def get_x(self):
        if self.__corpus is None:
            self.__corpus = self.__read_corpus()

        return self.__corpus[self.__x_column_name]

    def get_y(self):
        if self.__corpus is None:
            self.__corpus = self.__read_corpus()

        return self.__corpus[self.__y_column_name]

    def get_corpus(self) -> pd.DataFrame:
        return self.__corpus

    def save_corpus(self, corpus: pd.DataFrame):
        self.__corpus = corpus
        corpus.to_csv('../data/new/' + self.__corpus_name + '.csv', sep='|', index=False)

    def __read_corpus(self) -> pd.DataFrame:
        return pd.read_csv('../data/new/' + self.__corpus_name + '.csv', header=0, sep='|')
