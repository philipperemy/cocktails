import collections
import itertools
import math
import os
import re

import nltk
import nltk.classify.util
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

RT_POLARITY_POS_FILE = os.path.join('../data/pos.txt')
RT_POLARITY_NEG_FILE = os.path.join('../data/neg.txt')


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



# this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
    posFeatures = []
    negFeatures = []
    # http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    # breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWords = i.strip().split()
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWords = i.strip().split()
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)

    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 3/4))
    negCutoff = int(math.floor(len(negFeatures) * 3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    # trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)

    # initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    # puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    # prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    show_most_information_features(classifier, 30)


# creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])


# tries using all words as the feature selection mechanism
#print 'using all words as features'
#evaluate_features(make_full_dict)


# scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
    # creates lists of all positive and negative words
    posWords = []
    negWords = []
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWord = i.strip().split()
            posWords.append(posWord)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWord = i.strip().split()
            negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1

    # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    # builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


# finds word scores
word_scores = create_word_scores()


# finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


# creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

print 'using all words as features'
evaluate_features(make_full_dict)

# numbers of features to select
numbers_to_test = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 6000, 8000]
# tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print 'evaluating best %d word features' % (num)
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)
