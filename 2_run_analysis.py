from utils import *

# READ THIS: http://andybromberg.com/sentiment-analysis-python/

if __name__ == '__main__':
    # find word scores
    word_scores = create_word_scores()
    # numbers of features to select

    print('using all words as features')
    evaluate_features(make_full_dict, None)

    numbers_to_test = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 6000, 8000]
    # tries the best_word_features mechanism with each of the numbers_to_test of features
    for num in numbers_to_test:
        print('evaluating best %d word features' % (num))
        best_words = find_best_words(word_scores, num)
        evaluate_features(best_word_features, best_words)
        input("Press Enter to continue...")
