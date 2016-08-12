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

    from sklearn.ensemble import RandomForestRegressor

    cutoff = int(0.66 * len(ingredients_matrix))
    X_train = ingredients_matrix[:cutoff]
    Y_train = rating_values[:cutoff]

    X_test = ingredients_matrix[cutoff:]
    Y_test = rating_values[cutoff:]

    reg = RandomForestRegressor(n_estimators=30, min_samples_split=1, n_jobs=-1,
                                verbose=2, random_state=123, criterion='mse')
    reg.fit(X_train, Y_train)
    pred = reg.predict(X_test)
    for i in range(len(pred)):
        print('pred = {}, expected = {}'.format(pred[i], Y_test[i]))
    print('MAE = {}'.format(np.mean(np.abs(pred - Y_test))))
    print('MAE dummy = {}'.format(np.mean(np.abs(np.mean(Y_train) - Y_test))))
