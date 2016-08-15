from __future__ import print_function

from main import *


def write_to_file(name, line, decode=True):
    with open(name, 'a') as f:
        if decode:
            f.write(line.encode('utf-8'))
        else:
            f.write(line)
        f.write('\n')


if __name__ == '__main__':
    urls = get_cocktail_list()
    recipes = get_recipe_and_rating(urls)

    descriptions = []
    for recipe in recipes:
        descriptions.append('. '.join([v[2] for v in recipe[3]]).decode('utf-8'))
    labels = np.array([v[1] for v in recipes])
    mean_label = np.mean(labels.flatten())

    for i in range((len(labels))):
        if labels[i] > mean_label:
            write_to_file('good.txt', descriptions[i])
        else:
            write_to_file('bad.txt', descriptions[i])
        write_to_file('names.txt', recipes[i][0], decode=False)
