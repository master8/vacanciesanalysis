from datetime import datetime

import pandas as pd
import pymystem3

from classification.experiments.logreg import LogisticRegressionExperiments
from classification.tokenization import Tokenizer, TokensProvider
from classification.vectorization import Vectorizer, VectorsProvider

# pymystem3.mystem.MYSTEM_DIR = "/home/mluser/anaconda3/envs/master8_env/.local/bin"
# pymystem3.mystem.MYSTEM_BIN = "/home/mluser/anaconda3/envs/master8_env/.local/bin/mystem"


# Tokenizer().tokenize()

vectorizer = Vectorizer()
# vectorizer.vectorize_with_tfidf()
vectorizer.vectorize_with_w2v()


logreg = LogisticRegressionExperiments()
logreg.make_use_tfidf()