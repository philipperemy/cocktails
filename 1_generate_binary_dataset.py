import numpy as np

from utils import *

if __name__ == '__main__':
    recipes = get_recipe_and_rating(get_cocktail_list())

    descriptions = []
    for recipe in recipes:
        descriptions.append('. '.join([v[2] for v in recipe[3]]))
    labels = np.array([v[1] for v in recipes])
    mean_label = np.mean(labels.flatten())

    remove_if_any(POS_FILE)
    remove_if_any(NEG_FILE)

    for i in range((len(labels))):
        if labels[i] > mean_label:
            write_to_file(POS_FILE, descriptions[i])
        else:
            write_to_file(NEG_FILE, descriptions[i])

    for filename in [POS_FILE, NEG_FILE]:
        if TUPLE_LENGTH > 1:
            read_and_write(filename, filename + '.w', tuple_length=TUPLE_LENGTH)
            os.rename(filename + '.w', filename)
