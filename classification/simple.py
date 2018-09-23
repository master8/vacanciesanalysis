import random
import string

import numpy as np
import pandas as pd

from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

import warnings


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


def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False):
    """Plot a confusion matrix via matplotlib.
    Parameters
    -----------
    conf_mat : array-like, shape = [n_classes, n_classes]
        Confusion matrix from evaluate.confusion matrix.
    hide_spines : bool (default: False)
        Hides axis spines if True.
    hide_ticks : bool (default: False)
        Hides axis ticks if True
    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure
    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.Blues if `None`
    colorbar : bool (default: False)
        Shows a colorbar if True
    show_absolute : bool (default: True)
        Shows absolute confusion matrix coefficients if True.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    show_normed : bool (default: False)
        Shows normed confusion matrix coefficients if True.
        The normed confusion matrix coefficients give the
        proportion of training examples per class that are
        assigned the correct label.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
    """
    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat)*1.25, len(conf_mat)*1.25)

    if show_absolute:
        matshow = ax.matshow(conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                # cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            # else:
                # cell_text += format(normed_conf_mat[i, j], '.2f')
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax


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


matrix = confusion_matrix(y_true, y_pred)
fig, ax = plot_confusion_matrix(matrix, colorbar=True, show_absolute=False, show_normed=True)
plt.xticks(np.arange(0, 12, 1))
plt.yticks(np.arange(0, 12, 1))
plt.show()


x_train_vec = np.zeros((len(x_train), 300))
for i in range(0, len(x_train)):
    x_train_vec[i] = d2v_model.infer_vector(x_train[i].words, steps=3)

y_true, y_pred = test_classifier(d2v_model, classifier, x_train_vec, y_train)

matrix = confusion_matrix(y_true, y_pred)
fig, ax = plot_confusion_matrix(matrix, colorbar=True, show_absolute=False, show_normed=True)
plt.xticks(np.arange(0, 12, 1))
plt.yticks(np.arange(0, 12, 1))
plt.show()