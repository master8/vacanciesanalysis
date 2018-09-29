import random
import string

import numpy as np
import pandas as pd

from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

import warnings

from classification.visualisation.classification_results import show_classification_results


warnings.simplefilter(action='ignore', category=FutureWarning)


def read_dataset(path):
    dataset = pd.read_csv("../data/old/old_marked_vacancies_from_hh.csv", header=0)
    x_train, x_test, y_train, y_test = train_test_split(dataset.requirements,
                                                        dataset.profession,
                                                        random_state=0,
                                                        test_size=0.1)

    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        v = v.translate(str.maketrans("", "", string.punctuation))
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled


def train_doc2vec(corpus, epochs_size):
    d2v = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=3,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          epochs=epochs_size,
                          dm=1)  # dm defines the training algorithm. If dm=1 means ‘distributed memory’ (PV-DM)
                                 # and dm =0 means ‘distributed bag of words’ (PV-DBOW)
    d2v.build_vocab(corpus)

    # 10 epochs take around 10 minutes on my machine (i7), if you have more time/computational power make it 20
    # for epoch in range(10):
    d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
    # shuffle the corpus
    # random.shuffle(corpus)
    # decrease the learning rate
    # d2v.alpha -= 0.0002
    # fix the learning rate, no decay
    # d2v.min_alpha = d2v.alpha

    d2v.save("d2v.model")
    return d2v


def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_classifier(d2v, training_vectors, training_labels):
    train_vectors = get_vectors(d2v, len(training_vectors), 300, 'Train')
    model = LogisticRegression(C =0.5,solver = 'liblinear')
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)

    print("\nClassifier training")
    print('Training predicted classes: {}'.format(np.unique(training_predictions)))
    print('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    print('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    # test_vectors = get_vectors(d2v, len(testing_vectors), 300, 'Test')
    testing_predictions = classifier.predict(testing_vectors)

    print("\nClassifier testing")
    print('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    print('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    print('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))

    return testing_labels, testing_predictions


x_train, x_test, y_train, y_test, all_data = read_dataset("../data/old/old_marked_vacancies_from_hh.csv")

# for i in range(100, 150, 5):
#     print('\nepochs: ' + str(i))
#     d2v_model = train_doc2vec(all_data, i)
#     classifier = train_classifier(d2v_model, x_train, y_train)
#     test_classifier(d2v_model, classifier, x_test, y_test)

# d2v_model = train_doc2vec(x_train, 110)
d2v_model = doc2vec.Doc2Vec.load('d2v.model')
classifier = train_classifier(d2v_model, x_train, y_train)

x_test_vec = np.zeros((len(x_test), 300))
for i in range(0, len(x_test)):
    x_test_vec[i] = d2v_model.infer_vector(x_test[i].words, steps=3)

y_true, y_pred = test_classifier(d2v_model, classifier, x_test_vec, y_test)

show_classification_results(y_true, y_pred)


x_train_vec = np.zeros((len(x_train), 300))
for i in range(0, len(x_train)):
    x_train_vec[i] = d2v_model.infer_vector(x_train[i].words, steps=3)

y_true, y_pred = test_classifier(d2v_model, classifier, x_train_vec, y_train)

show_classification_results(y_true, y_pred)