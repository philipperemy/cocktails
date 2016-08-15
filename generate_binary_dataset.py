#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import subprocess

import numpy as np

from utils import *


def read_and_write(filename):
    lines = []
    with open(filename, 'r') as r:
        lines.extend(r.readlines())
    with open(filename + '.w', 'a') as w:
        for line in lines:
            w.write(filter_line(line))
            w.write('\n')


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

    os.remove('good.txt')
    os.remove('bad.txt')

    for i in range((len(labels))):
        if labels[i] > mean_label:
            write_to_file('good.txt', descriptions[i])
        else:
            write_to_file('bad.txt', descriptions[i])

    for filename in ['good.txt', 'bad.txt']:
        bash_command = "sed 's/[àâä]/a/g; s/[ÀÂÄ]/A/g; s/[éèêë]/e/g; s/[ÉÈÊË]/E/g; s/[îï]/i/g; s/[ÎÏ]/I/g; s/[ôö]/o/g; s/[ÖÔ]/O/g; s/[ûüù]/u/g; s/[ÛÜÙ]/U/g; s/ç/c/g; s/Ç/C/g' {0:} > {0:}.2".format(
            filename)
        print('executing {}'.format(bash_command))
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print(output)

        os.remove('{}.2.w'.format(filename))
        read_and_write('{}.2'.format(filename))

    os.remove('good.txt.2')
    os.remove('bad.txt.2')
