from __future__ import print_function

import math

import nltk
import nltk.classify.util
import nltk.metrics
from nltk.classify import NaiveBayesClassifier

from main import *


def show_most_information_features(instance, n):
    # Determine the most relevant features, and display them.
    cpdist = instance._feature_probdist
    print('Most Informative Features')

    for (fname, fval) in instance.most_informative_features(n):
        def labelprob(l):
            return cpdist[l, fname].prob(fval)

        labels = sorted([l for l in instance._labels
                         if fval in cpdist[l, fname].samples()],
                        key=labelprob)
        if len(labels) == 1:
            continue
        l0 = labels[0]
        l1 = labels[-1]
        if cpdist[l0, fname].prob(fval) == 0:
            ratio = 'INF'
        else:
            ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
                               cpdist[l0, fname].prob(fval))
        print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
               (fname.decode('utf-8'), fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))


def evaluate_features(feature_select_, reports_, labels_):
    features = []
    for i in range(len(reports_)):
        print(reports_[i])
        text = unicode(reports_[i])
        words = nltk.word_tokenize(text)
        words = [w.encode('UTF8').lower() for w in words]
        features.append([feature_select_(words), str(labels_[i])])  # feature_select is a function pointer.

    cutoff = int(math.floor(len(features) * 0.7))
    train_features = features[:cutoff]
    test_features = features[cutoff:]

    classifier = NaiveBayesClassifier.train(train_features)

    print('train on %d instances, test on %d instances' % (len(train_features), len(test_features)))
    print('accuracy:', nltk.classify.util.accuracy(classifier, test_features))
    show_most_information_features(classifier, 1000)


# creates a feature selection mechanism that uses all words
# try first with that.
def make_full_dict(words):
    return dict([(word, True) for word in words])


if __name__ == '__main__':
    urls = get_cocktail_list()
    recipes = get_recipe_and_rating(urls)

    descriptions = []
    for recipe in recipes:
        descriptions.append('. '.join(sum(recipe[3], [])).decode('utf-8'))
    labels = np.array([str(v[1]) for v in recipes])

    evaluate_features(make_full_dict, descriptions, labels)
