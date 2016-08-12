import collections

import numpy as np

from utils import *


def build_vocabulary(elements, max_features_percentage=1):
    c = collections.Counter(elements)
    max_features = int(max_features_percentage * len(c))
    features = dict()
    for i, element in enumerate(c.most_common()):
        element_name = element[0].decode('utf-8')
        if i < max_features:
            features[element_name] = i
        else:
            features[element_name] = max_features
    return features, max_features


def build_ingredient_matrix(recipes):
    ingredients = sum([[ingredient[2].strip().lower() for ingredient in recipe[3]] for recipe in recipes], [])
    features, max_features = build_vocabulary(ingredients)
    for recipe in recipes:
        recipe.append([features[ingredient[2].strip().lower().decode('utf-8')] for ingredient in recipe[3]])

    encoders = []
    for recipe in recipes:
        encoder = np.zeros((max_features + 1))
        for i in range(len(recipe[3])):
            quantity = recipe[3][i][0]
            if quantity is None or quantity == 'None':
                quantity = 1.0
            encoder[recipe[4][i]] = quantity
        encoders.append(encoder)
    return np.array(encoders)


if __name__ == '__main__':
    urls = get_cocktail_list()
    recipes = get_recipe_and_rating(urls)
    ingredients_matrix = build_ingredient_matrix(recipes)
    rating_values = np.array([v[1] for v in recipes])

    # np.mean(gnb.fit(ingredients_matrix, np.array(rating_values, dtype=str)).predict(ingredients_matrix) == np.array(rating_values, dtype=str))
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(ingredients_matrix, rating_values)
