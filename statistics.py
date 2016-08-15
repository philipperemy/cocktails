import matplotlib.pyplot as plt

from utils import *
import numpy as np
import scipy.stats as ss
if __name__ == '__main__':
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
