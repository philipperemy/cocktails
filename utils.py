#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import pickle
import re
from string import ascii_uppercase

# sys.setrecursionlimit(10000)

import requests
from bs4 import BeautifulSoup
from slugify import slugify


# TODO: no ingredients means REMOVE IT.
# concerns only two rows.
# Otherwise we're going to bias our model with ['Verre des dieux', 4.0/5.0, 4, []]

# recipeYield
# <span itemprop="recipeYield">pour 20 personnes</span>
# should be only for 1 PERSON.

def adjust_quantity_with_number_of_people(quantity, num_people):
    if quantity is None:
        return None
    if float(num_people) < 1:
        num_people = 1.0
    return float(quantity) / float(num_people)


def convert_to_float(number_str):
    # http://stackoverflow.com/questions/1806278/convert-fraction-to-float

    number_str = number_str.strip().lower().replace(',', '.').encode('ascii', 'ignore').replace('  ', '-')

    # recette-cocktail-maury-premier-printemps.html
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

        print(number_str)
        num, denom = number_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def clean_html(raw_html):
    clean_r = re.compile('<.*?>')
    clean_text = re.sub(clean_r, '', raw_html)
    return clean_text


def get_recipe_and_rating(urls, handler=None):
    recipes = []
    if not os.path.exists('recipes'):
        os.makedirs('recipes')
    for i, url in enumerate(urls):
        print('{}/{}'.format(i, len(urls)))
        cocktail_pickle = 'recipes/{}.pkl'.format(slugify(url))
        if os.path.isfile(cocktail_pickle):
            recipe = pickle.load(open(cocktail_pickle, 'r'))
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
            if group == None:
                for_how_many_people = 1
            else:
                for_how_many_people = float(group.group())
            structured_ingredients = []
            for content in ingredients:
                # if we have 1/4 => 0.25
                raw_quantity = convert_to_float(unicode(content[0].string))
                # we divide quantity per number of people.
                quantity = adjust_quantity_with_number_of_people(raw_quantity, for_how_many_people)
                unit = content[1].string
                ingredient = content[3].string
                structured_ingredients.append([quantity, unit, ingredient])
            structured_ingredients = [[unicode(e).encode('utf-8') for e in v] for v in structured_ingredients]
            recipe = [name, rating_value, rating_count, structured_ingredients]
            pickle.dump(recipe, open(cocktail_pickle, 'w'))
        print(recipe)
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
    cocktail_pickle = 'cocktail_set.pkl'
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


if __name__ == '__main__':
    urls = get_cocktail_list()
    recipes = get_recipe_and_rating(urls)
    print(recipes)
