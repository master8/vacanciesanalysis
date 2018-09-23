from classification.visualisation.classification_results import show_classification_results

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

model = KNeighborsClassifier(algorithm='auto', metric='minkowski', weights='distance')
