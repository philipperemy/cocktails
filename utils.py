import collections
import itertools
import math
import os
import pickle
import re
from string import ascii_uppercase

import nltk
import nltk.classify.util
import nltk.metrics
import requests
from bs4 import BeautifulSoup
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from slugify import slugify

NEG_FILE = 'data/neg.txt'
POS_FILE = 'data/pos.txt'
TUPLE_LENGTH = 10


def adjust_quantity_with_number_of_people(quantity, num_people):
    if quantity is None:
        return None
    if float(num_people) < 1:
        num_people = 1.0
    return float(quantity) / float(num_people)


def convert_to_float(number_str):
    # http://stackoverflow.com/questions/1806278/convert-fraction-to-float
    number_str = number_str.strip().lower().replace(',', '.').replace('  ', '-')
    if '%' in number_str:
        number_str = number_str.replace('%', '')
        return float(number_str) * 0.01

    if number_str == '':
        return None

    if '-' in number_str:  # 8-10 de lait.
        split_num = number_str.split('-')
        return 0.5 * float(split_num[1]) + 0.5 * float(split_num[0])
    if '=>' in number_str:  # cocktails/3579/recette-cocktail-antillaise.html
        split_num = number_str.split('=>')
        return 0.5 * float(split_num[1]) + 0.5 * float(split_num[0])
    try:
        return float(number_str)
    except ValueError:
        if 'un' in number_str:
            return 1.0
        if 'deux' in number_str:
            return 2.0
        if 'trois' in number_str:
            return 3.0
        if 'quatre' in number_str:
            return 4.0
        if 'cinq' in number_str:
            return 5.0
        if 'six' in number_str:
            return 6.0
        if 'sept' in number_str:
            return 7.0
        if 'huit' in number_str:
            return 8.0
        if 'neuf' in number_str:
            return 9.0
        if 'dix' in number_str:
            return 10.0
        if 'quelq' in number_str:
            return None  # arbitrary!
        if 'reste' in number_str:
            return None  # arbitrary !
        if number_str == 'n':
            return None

        # cocktails/3526/recette-cocktail-evain.html
        if 'ou' in number_str:
            split_num = number_str.split('ou')
            return 0.5 * float(split_num[1]) + 0.5 * float(split_num[0])

        if '½' in number_str:
            return float(number_str[0]) + 0.5

        if 'à' in number_str:
            return float(number_str.split(' ')[0])

        print(number_str)
        num, denom = number_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def get_recipe_and_rating(urls, handler=None):
    recipes = []
    if not os.path.exists('data/recipes'):
        os.makedirs('data/recipes')
    for i, url in enumerate(urls):
        if i % 1000 == 0:
            print('{}/{}'.format(i, len(urls)))
        cocktail_pickle = 'data/recipes/{}.pkl'.format(slugify(url))
        if os.path.isfile(cocktail_pickle):
            print('Found it ' + cocktail_pickle)
            recipe = pickle.load(open(cocktail_pickle, 'rb'))
            # recipe = None # - faster results but does not return anything.
        else:
            # factorize the code
            print(url)
            response = requests.get(u'http://www.1001cocktails.com/' + url)
            assert response.status_code == 200
            content = response.content
            soup = BeautifulSoup(content, 'html.parser')

            rating_value = float(soup.find_all('span', {'itemprop': 'ratingValue'})[0].next)
            rating_count = int(soup.find_all('span', {'itemprop': 'ratingCount'})[0].next)
            name = soup.find_all('h1', {'itemprop': 'name'})[0].next.encode('utf-8').strip()
            ingredients = [v.contents for v in soup.find_all('span', {'itemprop': 'ingredients'})]

            how_many_people_str = str(soup.find('span', {'itemprop': 'recipeYield'}).contents[0])
            group = re.search('[0-9]+', how_many_people_str)
            if group is None:
                for_how_many_people = 1
            else:
                for_how_many_people = float(group.group())
            structured_ingredients = []
            for content in ingredients:
                # if we have 1/4 => 0.25
                raw_quantity = convert_to_float(content[0].string)
                # we divide quantity per number of people.
                quantity = adjust_quantity_with_number_of_people(raw_quantity, for_how_many_people)
                unit = content[1].string
                ingredient = content[3].string
                structured_ingredients.append([quantity, unit, ingredient])
            structured_ingredients = [[str(e) for e in v] for v in structured_ingredients]
            recipe = [str(name), rating_value, rating_count, structured_ingredients]
            pickle.dump(recipe, open(cocktail_pickle, 'wb'))
        # print(recipe)
        if handler is not None:
            handler(recipe)
        else:
            recipes.append(recipe)
    return recipes


def get_links(url, depth=0):
    if depth > 1:
        return []
    print('To 1001cocktails -> {}'.format(url))
    links = []
    response = requests.get(url)
    assert response.status_code == 200
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    # prefix = 'http://www.1001cocktails.com/cocktails/'
    p = re.compile('cocktails/[0-9]{1,}/.+.html')
    for link in soup.find_all('a'):
        if 'href' in link.attrs:
            href = link.attrs['href']
            candidate = p.findall(href)
            if len(candidate) > 0:
                links.append(candidate)
            if 'cocktails-commencant-par' in href and 'cocktails/liste-cocktails' not in href:
                links.extend(get_links('http://www.1001cocktails.com/cocktails/' + str(href), depth=depth + 1))
    return links


def get_cocktail_list():
    cocktail_pickle = 'data/cocktail_set.pkl'
    if os.path.isfile(cocktail_pickle):
        with open(cocktail_pickle, 'rb') as f:
            cocktail_set = pickle.load(f)
    else:
        cocktail_set = set()
        urls = ['http://www.1001cocktails.com/cocktails/lister_cocktails.php3']
        for c in ascii_uppercase:
            urls.append('http://www.1001cocktails.com/cocktails/liste-cocktails-commencant-par-{}.html'.format(c))
        for url in urls:
            links = get_links(url)
            for link in links:
                cocktail_set.add(link[0])
            print('{} cocktails found so far.'.format(len(cocktail_set)))
        with open(cocktail_pickle, 'wb') as f:
            pickle.dump(cocktail_set, f)
    return cocktail_set


def filter_line_aux(line, tuple_length=2):
    elements = line.split('.')
    new_elements = []
    for elt in elements:
        new_element = []
        for split_elt in elt.split():
            split_elt = split_elt.strip()
            split_elt = split_elt.replace("'", '')
            if split_elt == 'de':
                continue
            new_element.append(split_elt)
        new_elements.append(new_element)
    new_str = ' '.join(['_'.join(v) for v in new_elements])
    return ' '.join(['+'.join(sorted(v)) for v in set(itertools.combinations(new_str.split(), tuple_length))])


def filter_line(line, tuple_length=2):
    out = ''
    for i in range(1, tuple_length + 1, 1):
        out += filter_line_aux(line, tuple_length=i) + ' '
    return out


def remove_if_any(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def show_statistics():
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as ss
    urls = get_cocktail_list()
    recipes = get_recipe_and_rating(urls)
    plt.hist([float(v[1]) for v in recipes], bins=20)
    plt.title('Distribution of the ratings')
    plt.ylabel('Ratings')
    plt.show()

    a = np.array([float(v[1]) for v in recipes])
    print(ss.kurtosis(a))
    print(ss.skew(a))
    print(np.mean(a))


def read_and_write(input_filename, output_filename, tuple_length=2):
    lines = []
    with open(input_filename, 'r') as r:
        lines.extend(r.readlines())
    with open(output_filename, 'w') as w:
        for line in lines:
            w.write(filter_line(line, tuple_length=tuple_length))
            w.write('\n')


def write_to_file(name, line):
    with open(name, 'a') as f:
        f.write(line)
        f.write('\n')
        f.flush()


# this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select, best_words):
    posFeatures = []
    negFeatures = []
    # http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    # breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    with open(POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWords = i.strip().split()
            posWords = [feature_select(posWords, best_words), 'pos']
            posFeatures.append(posWords)
    with open(NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWords = i.strip().split()
            negWords = [feature_select(negWords, best_words), 'neg']
            negFeatures.append(negWords)

    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
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
    print('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    classifier.show_most_informative_features(10)


# creates a feature selection mechanism that uses all words
def make_full_dict(words, not_used=None):
    return dict([(word, True) for word in words])


# scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
    # creates lists of all positive and negative words
    posWords = []
    negWords = []
    with open(POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWord = i.strip().split()
            posWords.append(posWord)
    with open(NEG_FILE, 'r') as negSentences:
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
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


# finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    # best_vals = sorted(word_scores.items(), key=lambda (w, s): s, reverse=True)[:number]
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


# creates feature selection mechanism that only uses best words
def best_word_features(words, best_words):
    return dict([(word, True) for word in words if word in best_words])


if __name__ == '__main__':
    print(filter_line('A. B. C.', tuple_length=3))
    print(filter_line('C. B. A.', tuple_length=3))
    # B_C A_B A_C - must be the same.
    # A_B A_C B_C
    # urls = get_cocktail_list()
    # recipes = get_recipe_and_rating(urls)
    # print(recipes)
