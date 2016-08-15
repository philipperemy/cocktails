#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import subprocess

import numpy as np

from utils import *

NEG_FILE = 'data/neg.txt'
POS_FILE = 'data/pos.txt'
TUPLE_LENGTH = 3


def read_and_write(filename, tuple_length=2):
    lines = []
    with open(filename, 'r') as r:
        lines.extend(r.readlines())
    with open(filename + '.w', 'a') as w:
        for line in lines:
            w.write(filter_line(line, tuple_length=tuple_length))
            w.write('\n')


def write_to_file(name, line, decode=True):
    with open(name, 'a') as f:
        if decode:
            f.write(line.encode('utf-8'))
        else:
            f.write(line)
        f.write('\n')


def remove_if_any(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


if __name__ == '__main__':
    urls = get_cocktail_list()
    recipes = get_recipe_and_rating(urls)

    descriptions = []
    for recipe in recipes:
        descriptions.append('. '.join([v[2] for v in recipe[3]]).decode('utf-8'))
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
        bash_command = "sed 's/[àâä]/a/g; s/[ÀÂÄ]/A/g; s/[éèêë]/e/g; s/[ÉÈÊË]/E/g; s/[îï]/i/g; s/[ÎÏ]/I/g; s/[ôö]/o/g; s/[ÖÔ]/O/g; s/[ûüù]/u/g; s/[ÛÜÙ]/U/g; s/ç/c/g; s/Ç/C/g; s/°/o/g; s/Ã©/e/g' {0:} > {0:}.2".format(
            filename)
        print('executing {}'.format(bash_command))
        process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
        output = process.communicate()[0]
        print(output)

        remove_if_any('{}.2.w'.format(filename))
        read_and_write('{}.2'.format(filename), tuple_length=TUPLE_LENGTH)
        remove_if_any('{}.2'.format(filename))
        os.rename('{}.2.w'.format(filename), filename)
